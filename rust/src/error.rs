//! Error types and FFI-compatible status codes.
//!
//! Each variant carries a fixed `i32` discriminant that is returned across the
//! C ABI boundary to the calling plugin layer.

use thiserror::Error;

/// Runtime error variants for the computational backend.
///
/// Represented as `#[repr(i32)]` so that the discriminant can be passed
/// directly through the C foreign-function interface.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TropError {
    /// No error (code 0).
    #[error("Success")]
    Success = 0,

    /// A null pointer was passed across the FFI boundary (code 1).
    #[error("Null pointer")]
    NullPointer = 1,

    /// Matrix or vector dimensions are incompatible (code 2).
    #[error("Invalid dimension")]
    InvalidDimension = 2,

    /// The dataset contains no control (untreated) units (code 3).
    #[error("No control units")]
    NoControl = 3,

    /// The dataset contains no treated units (code 4).
    #[error("No treated units")]
    NoTreated = 4,

    /// An iterative solver did not converge within the allowed iterations (code 5).
    #[error("Convergence failure")]
    Convergence = 5,

    /// A matrix required for inversion or decomposition is singular (code 6).
    #[error("Singular matrix")]
    Singular = 6,

    /// Heap allocation failed (code 7).
    #[error("Memory allocation failure")]
    Memory = 7,

    /// An unrecoverable panic was caught at the FFI boundary (code 8).
    #[error("Rust panic")]
    RustPanic = 8,

    /// Leave-one-out cross-validation could not complete (code 9).
    #[error("LOOCV failure")]
    LoocvFail = 9,

    /// Bootstrap variance estimation failed (code 10).
    #[error("Bootstrap failure")]
    BootstrapFail = 10,

    /// Unclassified numerical or logical error (code 11).
    #[error("Computation failure")]
    Computation = 11,

    /// FPC value is invalid: non-positive, non-finite, or less than the
    /// number of sampled PSUs in a stratum (code 12).
    #[error("Invalid FPC value")]
    InvalidFpc = 12,

    /// A singleton stratum (n_h=1) was encountered in strict mode where
    /// no lonely-PSU adjustment is allowed (code 13).
    #[error("Singleton PSU stratum with no adjustment")]
    SingletonPsu = 13,
}

impl TropError {
    /// Returns the `i32` discriminant for this variant.
    #[inline]
    pub fn code(&self) -> i32 {
        *self as i32
    }

    /// Maps an `i32` status code back to a variant.
    ///
    /// Returns `None` when `code` does not correspond to any defined variant.
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(TropError::Success),
            1 => Some(TropError::NullPointer),
            2 => Some(TropError::InvalidDimension),
            3 => Some(TropError::NoControl),
            4 => Some(TropError::NoTreated),
            5 => Some(TropError::Convergence),
            6 => Some(TropError::Singular),
            7 => Some(TropError::Memory),
            8 => Some(TropError::RustPanic),
            9 => Some(TropError::LoocvFail),
            10 => Some(TropError::BootstrapFail),
            11 => Some(TropError::Computation),
            12 => Some(TropError::InvalidFpc),
            13 => Some(TropError::SingletonPsu),
            _ => None,
        }
    }

    /// Returns `true` when the variant is [`Success`](TropError::Success).
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(self, TropError::Success)
    }
}

impl From<TropError> for i32 {
    #[inline]
    fn from(err: TropError) -> i32 {
        err.code()
    }
}

/// Convenience alias: `Result<T, TropError>`.
pub type TropResult<T> = Result<T, TropError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(TropError::Success.code(), 0);
        assert_eq!(TropError::NullPointer.code(), 1);
        assert_eq!(TropError::InvalidDimension.code(), 2);
        assert_eq!(TropError::NoControl.code(), 3);
        assert_eq!(TropError::NoTreated.code(), 4);
        assert_eq!(TropError::Convergence.code(), 5);
        assert_eq!(TropError::Singular.code(), 6);
        assert_eq!(TropError::Memory.code(), 7);
        assert_eq!(TropError::RustPanic.code(), 8);
        assert_eq!(TropError::LoocvFail.code(), 9);
        assert_eq!(TropError::BootstrapFail.code(), 10);
        assert_eq!(TropError::Computation.code(), 11);
        assert_eq!(TropError::InvalidFpc.code(), 12);
        assert_eq!(TropError::SingletonPsu.code(), 13);
    }

    #[test]
    fn test_from_code() {
        assert_eq!(TropError::from_code(0), Some(TropError::Success));
        assert_eq!(TropError::from_code(5), Some(TropError::Convergence));
        assert_eq!(TropError::from_code(12), Some(TropError::InvalidFpc));
        assert_eq!(TropError::from_code(13), Some(TropError::SingletonPsu));
        assert_eq!(TropError::from_code(99), None);
    }

    #[test]
    fn test_into_i32() {
        let err: i32 = TropError::Convergence.into();
        assert_eq!(err, 5);
    }

    #[test]
    fn test_is_success() {
        assert!(TropError::Success.is_success());
        assert!(!TropError::Convergence.is_success());
    }
}

#[cfg(test)]
mod error_propagation_tests {
    use super::*;
    use std::collections::HashSet;

    /// All defined TropError variants for exhaustive testing.
    fn all_variants() -> Vec<TropError> {
        vec![
            TropError::Success,
            TropError::NullPointer,
            TropError::InvalidDimension,
            TropError::NoControl,
            TropError::NoTreated,
            TropError::Convergence,
            TropError::Singular,
            TropError::Memory,
            TropError::RustPanic,
            TropError::LoocvFail,
            TropError::BootstrapFail,
            TropError::Computation,
            TropError::InvalidFpc,
            TropError::SingletonPsu,
        ]
    }

    /// Verify all error codes are unique (no two variants share the same i32 code).
    #[test]
    fn test_all_error_codes_unique() {
        let variants = all_variants();
        let mut seen_codes = HashSet::new();
        for v in &variants {
            let code = v.code();
            assert!(
                seen_codes.insert(code),
                "Duplicate error code {} for variant {:?}",
                code,
                v
            );
        }
    }

    /// Verify every variant has a mapping via from_code and the round-trip is consistent.
    #[test]
    fn test_error_code_mapping_complete() {
        let variants = all_variants();
        for v in &variants {
            let code = v.code();
            let recovered = TropError::from_code(code);
            assert_eq!(
                recovered,
                Some(*v),
                "from_code({}) should return {:?}, got {:?}",
                code,
                v,
                recovered
            );
        }
    }

    /// Verify that codes are contiguous from 0..=max and no gaps exist.
    #[test]
    fn test_error_codes_contiguous() {
        let variants = all_variants();
        let max_code = variants.iter().map(|v| v.code()).max().unwrap();
        for code in 0..=max_code {
            assert!(
                TropError::from_code(code).is_some(),
                "Gap in error codes: code {} has no corresponding variant",
                code
            );
        }
    }

    /// Verify every variant's Display output is non-empty.
    #[test]
    fn test_error_display_non_empty() {
        let variants = all_variants();
        for v in &variants {
            let display = format!("{}", v);
            assert!(
                !display.is_empty(),
                "Display for {:?} should not be empty",
                v
            );
        }
    }

    /// Verify every variant's Debug output is non-empty.
    #[test]
    fn test_error_debug_non_empty() {
        let variants = all_variants();
        for v in &variants {
            let debug = format!("{:?}", v);
            assert!(
                !debug.is_empty(),
                "Debug for {:?} should not be empty",
                v
            );
        }
    }

    /// Verify from_code returns None for invalid codes.
    #[test]
    fn test_error_invalid_codes() {
        assert_eq!(TropError::from_code(-1), None);
        assert_eq!(TropError::from_code(14), None);
        assert_eq!(TropError::from_code(100), None);
        assert_eq!(TropError::from_code(i32::MAX), None);
        assert_eq!(TropError::from_code(i32::MIN), None);
    }

    /// Verify the Into<i32> trait implementation matches code().
    #[test]
    fn test_into_i32_matches_code() {
        let variants = all_variants();
        for v in &variants {
            let code_method = v.code();
            let code_into: i32 = (*v).into();
            assert_eq!(
                code_method, code_into,
                "code() and Into<i32> disagree for {:?}: {} vs {}",
                v, code_method, code_into
            );
        }
    }

    /// Verify is_success is only true for the Success variant.
    #[test]
    fn test_is_success_only_for_success() {
        let variants = all_variants();
        for v in &variants {
            if *v == TropError::Success {
                assert!(v.is_success(), "Success should report is_success=true");
            } else {
                assert!(!v.is_success(), "{:?} should report is_success=false", v);
            }
        }
    }

    /// Verify non-zero error codes for all non-Success variants (FFI contract).
    #[test]
    fn test_non_success_codes_nonzero() {
        let variants = all_variants();
        for v in &variants {
            if *v != TropError::Success {
                assert_ne!(
                    v.code(),
                    0,
                    "Non-success variant {:?} must have non-zero code",
                    v
                );
            }
        }
    }
}
