use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::codegen::RegBudget;
use crate::ir::CompileError;

pub const DEFAULT_AMD_ZEN_EPYC_SPEC: &str = include_str!("../specs/amd_zen_epyc.spec");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimdTier {
    Avx2Split128,
    Avx2,
    Avx512Split256,
    Avx512Native512,
}

impl SimdTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Avx2Split128 => "avx2_128x2",
            Self::Avx2 => "avx2",
            Self::Avx512Split256 => "avx512_256x2",
            Self::Avx512Native512 => "avx512_512",
        }
    }

    pub fn parse_spec(s: &str) -> Result<Self, CompileError> {
        match s.trim() {
            "avx2_128x2" => Ok(Self::Avx2Split128),
            "avx2" => Ok(Self::Avx2),
            "avx512_256x2" => Ok(Self::Avx512Split256),
            "avx512_512" => Ok(Self::Avx512Native512),
            other => Err(CompileError::ParseError(format!("unknown simd tier '{other}'"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelRange {
    pub lo: u16,
    pub hi: u16,
}

impl ModelRange {
    pub const fn contains(self, model: u16) -> bool {
        model >= self.lo && model <= self.hi
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuProfile {
    pub name: String,
    pub family: u16,
    pub models: Vec<ModelRange>,
    pub l1d_kib: usize,
    pub l1d_ways: usize,
    pub l2_kib: usize,
    pub l3_ccx_mib: usize,
    pub l3_3dv_mib: usize,
    pub ccx_cores: usize,
    pub line_bytes: usize,
    pub simd: SimdTier,
    pub vec_regs: usize,
    pub vec_bits: usize,
    pub native_pipe_bits: usize,
}

impl CpuProfile {
    pub fn matches_cpuid(&self, family: u16, model: u16) -> bool {
        self.family == family && self.models.iter().any(|r| r.contains(model))
    }

    pub const fn reg_budget(&self) -> RegBudget {
        RegBudget { max_gprs: 14, max_vecs: self.vec_regs }
    }

    pub const fn l1_bench_tile_bytes(&self) -> usize {
        if self.l1d_kib == 0 {
            16 * 1024
        } else {
            (self.l1d_kib * 1024) / 2
        }
    }

    pub const fn prefetch_distance_bytes(&self) -> usize {
        let line = if self.line_bytes == 0 { 64 } else { self.line_bytes };
        line.saturating_mul(4)
    }

    pub const fn streaming_ram_tile_bytes(&self) -> usize {
        4 * 1024 * 1024
    }

    pub const fn supports_avx512(&self) -> bool {
        self.vec_bits >= 512 && self.vec_regs >= 32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostCpuInfo {
    pub vendor: String,
    pub family: u16,
    pub model: u16,
    pub has_avx2: bool,
    pub has_avx512f: bool,
}

fn perr(msg: impl Into<String>) -> CompileError {
    CompileError::ParseError(msg.into())
}

fn parse_models(raw: &str) -> Result<Vec<ModelRange>, CompileError> {
    let mut out = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let range = if let Some((a, b)) = part.split_once("..=").or_else(|| part.split_once("..")) {
            let lo: u16 = a.trim().parse().map_err(|_| perr(format!("bad model lo '{part}'")))?;
            let hi: u16 = b.trim().parse().map_err(|_| perr(format!("bad model hi '{part}'")))?;
            if lo > hi {
                return Err(perr(format!("inverted model range '{part}'")));
            }
            ModelRange { lo, hi }
        } else {
            let m: u16 = part.parse().map_err(|_| perr(format!("bad model '{part}'")))?;
            ModelRange { lo: m, hi: m }
        };
        out.push(range);
    }
    if out.is_empty() {
        return Err(perr("empty models list"));
    }
    Ok(out)
}

fn req_str<'a>(map: &'a BTreeMap<String, String>, sec: &str, key: &str) -> Result<&'a str, CompileError> {
    map.get(key)
        .map(String::as_str)
        .ok_or_else(|| perr(format!("section [{sec}] missing '{key}'")))
}

fn req_usize(map: &BTreeMap<String, String>, sec: &str, key: &str) -> Result<usize, CompileError> {
    let raw = req_str(map, sec, key)?;
    raw.parse::<usize>()
        .map_err(|_| perr(format!("section [{sec}] invalid integer for '{key}': '{raw}'")))
}

fn finish_section(name: String, kv: &BTreeMap<String, String>) -> Result<CpuProfile, CompileError> {
    let fam_u = req_usize(kv, &name, "family")?;
    let family = u16::try_from(fam_u).map_err(|_| perr(format!("section [{name}] family overflow")))?;
    let models = parse_models(req_str(kv, &name, "models")?)?;
    let simd = SimdTier::parse_spec(req_str(kv, &name, "simd")?)?;
    let l1d_kib = req_usize(kv, &name, "l1d_kib")?;
    let l1d_ways = req_usize(kv, &name, "l1d_ways")?;
    let l2_kib = req_usize(kv, &name, "l2_kib")?;
    let l3_ccx_mib = req_usize(kv, &name, "l3_ccx_mib")?;
    let l3_3dv_mib = req_usize(kv, &name, "l3_3dv_mib")?;
    let ccx_cores = req_usize(kv, &name, "ccx_cores")?;
    let line_bytes = req_usize(kv, &name, "line_bytes")?;
    let vec_regs = req_usize(kv, &name, "vec_regs")?;
    let vec_bits = req_usize(kv, &name, "vec_bits")?;
    let native_pipe_bits = req_usize(kv, &name, "native_pipe_bits")?;
    if l1d_kib == 0 || line_bytes == 0 || vec_regs == 0 || vec_bits == 0 {
        return Err(perr(format!("section [{name}] has zero architectural dimension")));
    }
    Ok(CpuProfile {
        name,
        family,
        models,
        l1d_kib,
        l1d_ways,
        l2_kib,
        l3_ccx_mib,
        l3_3dv_mib,
        ccx_cores,
        line_bytes,
        simd,
        vec_regs,
        vec_bits,
        native_pipe_bits,
    })
}

pub fn parse_cpu_specs(src: &str) -> Result<Vec<CpuProfile>, CompileError> {
    if !src.is_ascii() {
        return Err(perr("cpu spec must be pure 7-bit ASCII"));
    }
    let mut out = Vec::new();
    let mut cur_name: Option<String> = None;
    let mut cur_kv: BTreeMap<String, String> = BTreeMap::new();
    for (idx, raw_line) in src.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let sec = inner.trim();
            if sec.is_empty() || !sec.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
                return Err(perr(format!("line {}: invalid section header '{line}'", idx + 1)));
            }
            if let Some(prev) = cur_name.replace(sec.to_string()) {
                out.push(finish_section(prev, &cur_kv)?);
                cur_kv.clear();
            }
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            return Err(perr(format!("line {}: expected key=value, got '{line}'", idx + 1)));
        };
        if cur_name.is_none() {
            return Err(perr(format!("line {}: key=value before [section]", idx + 1)));
        }
        cur_kv.insert(k.trim().to_string(), v.trim().to_string());
    }
    if let Some(last) = cur_name {
        out.push(finish_section(last, &cur_kv)?);
    }
    if out.is_empty() {
        return Err(perr("cpu spec contains no profiles"));
    }
    Ok(out)
}

pub fn default_cpu_specs() -> Result<Vec<CpuProfile>, CompileError> {
    parse_cpu_specs(DEFAULT_AMD_ZEN_EPYC_SPEC)
}

pub fn load_cpu_specs_from_path(path: &Path) -> Result<Vec<CpuProfile>, CompileError> {
    let text = fs::read_to_string(path)
        .map_err(|e| perr(format!("read cpu spec '{}': {e}", path.display())))?;
    parse_cpu_specs(&text)
}

pub fn find_cpu_profile<'a>(specs: &'a [CpuProfile], name: &str) -> Option<&'a CpuProfile> {
    specs.iter().find(|p| p.name.eq_ignore_ascii_case(name.trim()))
}

pub fn parse_proc_cpuinfo(text: &str) -> Option<HostCpuInfo> {
    let (mut vendor, mut family, mut model) = (None, None, None);
    let (mut has_avx2, mut has_avx512f) = (false, false);
    for line in text.lines() {
        if line.trim().is_empty() && vendor.is_some() && family.is_some() && model.is_some() {
            break;
        }
        let Some((k, v)) = line.split_once(':') else { continue };
        let (kt, vt) = (k.trim(), v.trim());
        match kt {
            "vendor_id" => vendor = Some(vt.to_string()),
            "cpu family" => family = vt.parse::<u16>().ok(),
            "model" => model = vt.parse::<u16>().ok(),
            "flags" => {
                for fl in vt.split_whitespace() {
                    if fl == "avx2" {
                        has_avx2 = true;
                    } else if fl == "avx512f" {
                        has_avx512f = true;
                    }
                }
            }
            _ => {}
        }
    }
    Some(HostCpuInfo {
        vendor: vendor?,
        family: family?,
        model: model?,
        has_avx2,
        has_avx512f,
    })
}

pub fn parse_sysfs_cache_size_kib(text: &str) -> Option<usize> {
    let s = text.trim();
    if let Some(num) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
        num.trim().parse::<usize>().ok()
    } else if let Some(num) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        num.trim().parse::<usize>().ok().map(|m| m.saturating_mul(1024))
    } else {
        s.parse::<usize>().ok().map(|b| b / 1024)
    }
}

pub fn overlay_sysfs_cache(profile: &mut CpuProfile, cache_dir: &Path) {
    let Ok(entries) = fs::read_dir(cache_dir) else { return };
    let mut dirs: Vec<_> = entries
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("index")))
        .collect();
    dirs.sort();
    for dir in dirs {
        let level = fs::read_to_string(dir.join("level"))
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok());
        let ctype = fs::read_to_string(dir.join("type")).ok().map(|s| s.trim().to_string());
        let size_kib = fs::read_to_string(dir.join("size"))
            .ok()
            .and_then(|s| parse_sysfs_cache_size_kib(&s));
        let line = fs::read_to_string(dir.join("coherency_line_size"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());
        let ways = fs::read_to_string(dir.join("ways_of_associativity"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());
        match (level, ctype.as_deref()) {
            (Some(1), Some("Data")) => {
                if let Some(sz) = size_kib.filter(|&v| v > 0) {
                    profile.l1d_kib = sz;
                }
                if let Some(w) = ways.filter(|&v| v > 0) {
                    profile.l1d_ways = w;
                }
                if let Some(lb) = line.filter(|&v| v > 0) {
                    profile.line_bytes = lb;
                }
            }
            (Some(2), Some("Unified" | "Data")) => {
                if let Some(sz) = size_kib.filter(|&v| v > 0) {
                    profile.l2_kib = sz;
                }
            }
            (Some(3), Some("Unified")) => {
                if let Some(sz) = size_kib.filter(|&v| v >= 1024) {
                    let mib = sz / 1024;
                    if mib > 32 && profile.l3_3dv_mib > 0 {
                        profile.l3_3dv_mib = mib;
                    } else {
                        profile.l3_ccx_mib = mib;
                    }
                }
            }
            _ => {}
        }
    }
}

fn generic_fallback_profile(has_avx512f: bool) -> CpuProfile {
    if has_avx512f {
        CpuProfile {
            name: "generic_avx512".into(),
            family: 0,
            models: Vec::new(),
            l1d_kib: 32,
            l1d_ways: 8,
            l2_kib: 1024,
            l3_ccx_mib: 32,
            l3_3dv_mib: 0,
            ccx_cores: 8,
            line_bytes: 64,
            simd: SimdTier::Avx512Split256,
            vec_regs: 32,
            vec_bits: 512,
            native_pipe_bits: 256,
        }
    } else {
        CpuProfile {
            name: "generic_avx2".into(),
            family: 0,
            models: Vec::new(),
            l1d_kib: 32,
            l1d_ways: 8,
            l2_kib: 512,
            l3_ccx_mib: 16,
            l3_3dv_mib: 0,
            ccx_cores: 4,
            line_bytes: 64,
            simd: SimdTier::Avx2,
            vec_regs: 16,
            vec_bits: 256,
            native_pipe_bits: 256,
        }
    }
}

pub fn select_profile_for_host(specs: &[CpuProfile], info: &HostCpuInfo) -> CpuProfile {
    if info.vendor == "AuthenticAMD" {
        if let Some(p) = specs.iter().find(|p| p.matches_cpuid(info.family, info.model)) {
            return p.clone();
        }
        if let Some(p) = specs
            .iter()
            .find(|p| p.family == info.family && p.supports_avx512() == info.has_avx512f)
        {
            return p.clone();
        }
        if let Some(p) = specs.iter().find(|p| p.family == info.family) {
            return p.clone();
        }
        if let Some(p) = specs.iter().find(|p| p.supports_avx512() == info.has_avx512f) {
            return p.clone();
        }
    }
    generic_fallback_profile(info.has_avx512f)
}

pub fn detect_host_profile(specs: &[CpuProfile]) -> CpuProfile {
    let mut profile = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|txt| parse_proc_cpuinfo(&txt))
        .map(|info| select_profile_for_host(specs, &info))
        .unwrap_or_else(|| {
            find_cpu_profile(specs, "zen2_rome")
                .cloned()
                .unwrap_or_else(|| generic_fallback_profile(false))
        });
    overlay_sysfs_cache(&mut profile, Path::new("/sys/devices/system/cpu/cpu0/cache"));
    profile
}

pub fn resolve_cpu_profile(
    cpu_arg: Option<&str>,
    spec_path: Option<&Path>,
) -> Result<CpuProfile, CompileError> {
    let specs = match spec_path {
        Some(p) => load_cpu_specs_from_path(p)?,
        None => default_cpu_specs()?,
    };
    match cpu_arg.map(str::trim) {
        None | Some("") | Some("auto") => Ok(detect_host_profile(&specs)),
        Some(name) => find_cpu_profile(&specs, name).cloned().ok_or_else(|| {
            let names: Vec<&str> = specs.iter().map(|p| p.name.as_str()).collect();
            perr(format!("unknown cpu profile '{name}' (available: auto, {})", names.join(", ")))
        }),
    }
}
