#![forbid(unsafe_code)]

use std::process::Command;
use zero_cc::{
    compile_kernel_direct, fuse_kernels, optimize_kernel, parse_rust_kernel, parse_rust_pipeline,
    verify_elf64_object, RegBudget, TargetIsa,
};

const VMM_PIPELINE_SRC: &str = r#"
fn stage1_virtio_slot(desc: u64) -> u64 {
    (desc >> 48) & 255
}

fn stage2_gpa_translate(gpa: u64, len: u64) -> (u64, u64) {
    let end = gpa + len;
    let low_ok = if gpa < 0xC000_0000 { if end <= 0xC000_0000 { 1 } else { 0 } } else { 0 };
    let high_ok = if gpa >= 0x1_0000_0000 {
        if gpa < 0x2_0000_0000 {
            if end <= 0x2_0000_0000 { 1 } else { 0 }
        } else { 0 }
    } else { 0 };
    let host_off = if high_ok != 0 { gpa - 0x4000_0000 } else { gpa };
    (host_off, low_ok | high_ok)
}

fn stage3_netd_admit(src_ip: u64, cidr_net: u64, prefix: u64) -> u64 {
    let valid_prefix = if prefix <= 32 { 1 } else { 0 };
    let shift = if valid_prefix != 0 { 32 - prefix } else { 32 };
    let mask = (0xFFFF_FFFF << shift) & 0xFFFF_FFFF;
    let cidr_ok = if ((src_ip ^ cidr_net) & mask) == 0 { 1 } else { 0 };
    let not_meta = if src_ip != 0xA9FE_A9FE { 1 } else { 0 };
    valid_prefix & cidr_ok & not_meta
}

fn stage4_mtu_credits(len: u64) -> u64 {
    (len + 1499) / 1500
}

pub fn vmm_vmm_4stage(
    in_gpa: &[u64; 1],
    in_desc: &[u64; 1],
    in_net: &[u64; 1],
    in_prefix: &[u8; 1],
    out_res: &mut [u64; 2],
) {
    let net = in_net[0];
    let prefix = in_prefix[0] as u64;
    let net_ok = stage3_netd_admit(net & 0xFFFF_FFFF, net >> 32, prefix);

    let desc = in_desc[0];
    let len = desc & 0xFFFF_FFFF;
    let pre_ok = net_ok & (if ((desc >> 32) & 0xFFFC) == 0 { 1 } else { 0 }) & (if len > 0 { 1 } else { 0 });
    let slot = stage1_virtio_slot(desc);
    let credits = stage4_mtu_credits(len);
    let packed = 0x8000_0000_0000_0000 | (slot << 48) | (credits << 32) | len;

    let gpa = in_gpa[0];
    let (host_off, gpa_ok) = stage2_gpa_translate(gpa, len);
    let admitted = pre_ok & gpa_ok;

    out_res[0] = if admitted != 0 { host_off } else { 0 };
    out_res[1] = if admitted != 0 { packed } else { 0 };
}
"#;

fn vmm_oracle(gpa: u64, desc: u64, net: u64, prefix: u8) -> [u64; 2] {
    let len = desc & 0xFFFF_FFFF;
    let flags = (desc >> 32) & 0xFFFF;
    let slot = (desc >> 48) & 255;

    let (host_off, gpa_ok) = if let Some(end) = gpa.checked_add(len) {
        if len > 0 && gpa < 0xC000_0000 && end <= 0xC000_0000 {
            (gpa, 1u64)
        } else if len > 0 && gpa >= 0x1_0000_0000 && end <= 0x2_0000_0000 {
            (gpa - 0x4000_0000, 1u64)
        } else {
            (0u64, 0u64)
        }
    } else {
        (0u64, 0u64)
    };

    let src_ip = net & 0xFFFF_FFFF;
    let cidr_net = net >> 32;
    let valid_prefix = u64::from(prefix <= 32);
    let shift = if prefix <= 32 { 32 - u64::from(prefix) } else { 32 };
    let mask = (0xFFFF_FFFFu64 << shift) & 0xFFFF_FFFF;
    let cidr_ok = u64::from(((src_ip ^ cidr_net) & mask) == 0);
    let not_meta = u64::from(src_ip != 0xA9FE_A9FE);
    let flags_ok = u64::from((flags & 0xFFFC) == 0);
    let admitted = gpa_ok & valid_prefix & cidr_ok & not_meta & flags_ok;
    let credits = len.div_ceil(1500);

    if admitted != 0 {
        [host_off, (1u64 << 63) | (slot << 48) | (credits << 32) | len]
    } else {
        [0, 0]
    }
}

#[test]
fn vmm_vmm_4stage_pipeline_fuse_barrett_objdump_and_parity() {
    let parsed = parse_rust_pipeline(VMM_PIPELINE_SRC, "vmm_vmm_4stage").expect("parse vmm_vmm_4stage");
    let opt = optimize_kernel(&parsed).expect("optimize vmm_vmm_4stage");
    assert_eq!(opt.cost().divs_rems, 0, "Stage 4 MTU / 1500 must rewrite to Barrett multiply");

    // Verify fuse_kernels on a 2-stage split of the same pipeline.
    let s1 = parse_rust_kernel(
        "pub fn s1(in_gpa: &[u64; 1], in_desc: &[u64; 1], mid: &mut [u64; 2]) {
            let gpa = in_gpa[0];
            let len = in_desc[0] & 0xFFFF_FFFF;
            let end = gpa + len;
            let low_ok = if gpa < 0xC000_0000 { if end <= 0xC000_0000 { 1 } else { 0 } } else { 0 };
            let high_ok = if gpa >= 0x1_0000_0000 { if gpa < 0x2_0000_0000 { if end <= 0x2_0000_0000 { 1 } else { 0 } } else { 0 } } else { 0 };
            mid[0] = if high_ok != 0 { gpa - 0x4000_0000 } else { gpa };
            mid[1] = (if len > 0 { 1 } else { 0 }) & (low_ok | high_ok);
        }",
    )
    .expect("parse s1");
    let s2 = parse_rust_kernel(
        "pub fn s2(mid: &[u64; 2], in_desc: &[u64; 1], out_res: &mut [u64; 2]) {
            let len = in_desc[0] & 0xFFFF_FFFF;
            let credits = len.div_ceil(1500);
            let ok = mid[1];
            out_res[0] = if ok != 0 { mid[0] } else { 0 };
            out_res[1] = if ok != 0 { (credits << 32) | len } else { 0 };
        }",
    )
    .expect("parse s2");
    let fused = fuse_kernels("vmm_fused_2stage", &[&s1, &s2]).expect("fuse_kernels");
    assert_eq!(fused.bufs.len(), 3, "internal `mid` buffer must fuse away into SSA registers");
    let fused_opt = optimize_kernel(&fused).expect("optimize fused");
    assert_eq!(fused_opt.cost().divs_rems, 0);

    // Exhaustive adversarial IR parity across edge cases.
    let cases: &[(u64, u64, u64, u8)] = &[
        (0x0000_1000, 1500 | (1 << 32) | (7 << 48), 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0xBFFF_FF00, 512 | (3 << 48), 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0xC000_0000, 64 | (1 << 48), 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x1_0000_0000, 1501 | (3 << 32) | (255 << 48), 0x0A00_0005 | (0x0A00_00FF << 32), 24),
        (0x1_FFFF_FF00, 512 | (9 << 48), 0x0A00_0005 | (0x0A00_0000 << 32), 0),
        (u64::MAX, 1, 0x0A00_0001 | (0x0A00_0001 << 32), 32),
        (0x8000_0000_0000_0000, 64, 0x0A00_0001 | (0x0A00_0001 << 32), 32),
        (0x7FFF_FFFF_FFFF_FF00, 512, 0x0A00_0001 | (0x0A00_0001 << 32), 32),
        (0x1000, 0xFFFF_FFFF, 0x0A00_0001 | (0x0A00_00FF << 32), 0),
        (0x1000, 1499, 0xA9FE_A9FE | (0xA9FE_A900 << 32), 24),
        (0x1000, 1500, 0x0A00_0001 | (0x0A00_0001 << 32), 33),
        (0x1000, 1500 | (4 << 32), 0x0A00_0001 | (0x0A00_0001 << 32), 32),
        (0x1000, 0, 0x0A00_0001 | (0x0A00_0001 << 32), 32),
        // Six adversarial boundary cases: exact High-RAM straddle, 32-bit borrow flip,
        // len + 1499 u32 wrap boundary, and 8 distinct prefixes [0,8,16,20,24,28,32,33].
        (0x1_0000_0001, 0xFFFF_FFFF, 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x1_0000_0002, 0xFFFF_FFFF, 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x1_3FFF_FFFF, 64, 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x1_4000_0000, 64, 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x1000, 0xFFFF_FA24, 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x1000, 0xFFFF_FA25, 0x0A00_0001 | (0x0A00_0000 << 32), 24),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_00FF << 32), 0),
        (0x2000, 512, 0x0A00_0001 | (0x0A11_2233 << 32), 8),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_2233 << 32), 16),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_0F33 << 32), 20),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_0033 << 32), 24),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_000F << 32), 28),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_0001 << 32), 32),
        (0x2000, 512, 0x0A00_0001 | (0x0A00_0001 << 32), 33),
    ];
    for &(gpa, desc, net, prefix) in cases {
        let want = vmm_oracle(gpa, desc, net, prefix);
        let got_ir = opt
            .eval(&[&[gpa as i64], &[desc as i64], &[net as i64], &[i64::from(prefix)], &[0, 0]])
            .expect("eval opt");
        assert_eq!([got_ir[4][0] as u64, got_ir[4][1] as u64], want, "IR mismatch on ({gpa:#x}, {desc:#x}, {net:#x}, {prefix})");
    }
    // Verify exact oracle outcomes for the 4 key arithmetic boundaries.
    assert_eq!(vmm_oracle(0x1_0000_0001, 0xFFFF_FFFF, 0x0A00_0001 | (0x0A00_0000 << 32), 24), [0xC000_0001, (1u64 << 63) | (2_863_312u64 << 32) | 0xFFFF_FFFF]);
    assert_eq!(vmm_oracle(0x1_0000_0002, 0xFFFF_FFFF, 0x0A00_0001 | (0x0A00_0000 << 32), 24), [0, 0]);
    assert_eq!(vmm_oracle(0x1_3FFF_FFFF, 64, 0x0A00_0001 | (0x0A00_0000 << 32), 24)[0], 0x0_FFFF_FFFF);
    assert_eq!(vmm_oracle(0x1_4000_0000, 64, 0x0A00_0001 | (0x0A00_0000 << 32), 24)[0], 0x1_0000_0000);

    // Counterfactual non-VMM kernel with identical (u64, u64, u64, u8) -> [u64; 2] buffer shape:
    // fast-path superblock specialization must never miscompile a different output formula.
    let counter_k = parse_rust_kernel(
        "pub fn counter_vmm(in_gpa: &[u64; 1], in_desc: &[u64; 1], in_net: &[u64; 1], in_prefix: &[u8; 1], out_res: &mut [u64; 2]) {
            let len = in_desc[0] & 0xFFFF_FFFF;
            let credits = (len + 1499) / 1500;
            let ok = if in_gpa[0] < 0xC000_0000 { if (in_prefix[0] as u64) <= 32 { 1 } else { 0 } } else { 0 };
            out_res[0] = if ok != 0 { in_gpa[0] + 7 } else { 0 };
            out_res[1] = if ok != 0 { (credits << 32) | (in_net[0] & 0xFFFF_FFFF) } else { 0 };
        }",
    )
    .expect("parse counter_vmm");
    let counter_obj = compile_kernel_direct(&counter_k, RegBudget { max_gprs: 14, max_vecs: 32 }, TargetIsa::X86_64)
        .expect("compile counter_vmm");
    verify_elf64_object(&counter_obj.elf_bytes).expect("verify counter_vmm elf");
    assert!(counter_obj.zero_spills);

    // Compile direct ELF64 x86-64 + AArch64 and inspect disassembly.
    let budget = RegBudget { max_gprs: 14, max_vecs: 32 };
    let x86 = compile_kernel_direct(&parsed, budget, TargetIsa::X86_64).expect("compile x86_64 vmm");
    let a64 = compile_kernel_direct(&parsed, budget, TargetIsa::Aarch64).expect("compile aarch64 vmm");
    verify_elf64_object(&x86.elf_bytes).expect("verify x86_64 elf");
    verify_elf64_object(&a64.elf_bytes).expect("verify aarch64 elf");
    assert!(x86.zero_spills);

    #[cfg(not(miri))]
    {
        let obj_path = std::env::temp_dir().join("zero_cc_vmm_vmm_4stage.o");
        std::fs::write(&obj_path, &x86.elf_bytes).expect("write vmm .o");
        let dis = Command::new("objdump")
            .args(["-d", "-M", "intel"])
            .arg(&obj_path)
            .output()
            .expect("objdump vmm .o");
        let _ = std::fs::remove_file(&obj_path);
        assert!(dis.status.success());
        let asm = String::from_utf8_lossy(&dis.stdout);
        let mut ymm_cnt = 0usize;
        let mut div_cnt = 0usize;
        let mut spill_cnt = 0usize;
        let mut vptest_cnt = 0usize;
        let mut in_fn = false;
        for line in asm.lines() {
            if line.contains("<zero_cc_vmm_vmm_4stage_batch>:") { in_fn = true; continue; }
            if !in_fn { continue; }
            if line.contains("ymm") { ymm_cnt += 1; }
            if line.contains("vptest") { vptest_cnt += 1; }
            if line.contains("\tdiv ") || line.contains("\tidiv ") { div_cnt += 1; }
            if line.contains("[rsp") || line.contains("[rbp") { spill_cnt += 1; }
        }
        eprintln!("VMM_OBJ|insns={}|ymm={}|divs={}|spills={}|vptest={}|bytes={}", x86.direct_insns, ymm_cnt, div_cnt, spill_cnt, vptest_cnt, x86.elf_bytes.len());
        assert_eq!(div_cnt, 0, "found div/idiv in zero_cc_vmm_vmm_4stage_batch:\n{asm}");
        assert_eq!(spill_cnt, 0, "found stack spill in zero_cc_vmm_vmm_4stage_batch:\n{asm}");
        assert!(ymm_cnt > 100, "expected AVX2 Batch8Plan YMM emission, got ymm={ymm_cnt}:\n{asm}");
        assert!(vptest_cnt >= 1, "expected Pass-2 vptest superblock in zero_cc_vmm_vmm_4stage_batch:\n{asm}");

        let c_path = std::env::temp_dir().join("zero_cc_counter_vmm.o");
        std::fs::write(&c_path, &counter_obj.elf_bytes).expect("write counter_vmm .o");
        let c_dis = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&c_path).output().expect("objdump counter_vmm .o");
        let _ = std::fs::remove_file(&c_path);
        assert!(c_dis.status.success());
        let c_asm = String::from_utf8_lossy(&c_dis.stdout);
        assert!(!c_asm.contains("vptest"), "counterfactual counter_vmm must not trigger VMM vptest fast path:\n{c_asm}");
    }
}
