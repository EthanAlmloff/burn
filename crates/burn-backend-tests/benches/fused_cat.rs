//! Measures fused concatenation, including host preparation and device completion.
//!
//! Targets are total output element counts. Minimum segment sizes and alignment can
//! make small outputs exceed the target when concatenating many inputs.
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-backend-tests --bench fused_cat --features cuda,fusion
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};
use burn_tensor::{Device, StreamId, Tensor, TensorData};
use divan::Bencher;

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    println!("Fused concatenation benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Equal,
    Repeating,
    LargeFirst,
    LargeLast,
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    WholeVector,
    AlignedAxis,
    UnalignedAxis,
}

/// Inputs and output have rank two. `dim` joins segments; the other axis preserves its length.
fn bench_cat(
    bencher: Bencher,
    count: usize,
    pattern: Pattern,
    layout: Layout,
    target_elements: usize,
) {
    let mut lengths: Vec<usize> = (0..count)
        .map(|i| match pattern {
            Pattern::Equal => 4,
            Pattern::Repeating => [1, 2, 5, 6, 3][i % 5],
            Pattern::LargeFirst if i == 0 => count * 8 + 1,
            Pattern::LargeLast if i == count - 1 => count * 8 + 1,
            _ => 1,
        })
        .collect();
    match layout {
        Layout::WholeVector => {}
        Layout::AlignedAxis => {
            for length in &mut lengths {
                *length *= 4;
            }
        }
        Layout::UnalignedAxis => {
            // Keep the total aligned while leaving internal boundaries inside vectors.
            let total: usize = lengths.iter().sum();
            lengths[count - 1] += (4 - total % 4) % 4;
        }
    }
    let total: usize = lengths.iter().sum();
    let other = if matches!(layout, Layout::WholeVector) {
        (target_elements / total / 4).max(1) * 4
    } else {
        (target_elements / total).max(1)
    };
    let dim = usize::from(!matches!(layout, Layout::WholeVector));
    let stream = StreamId::allocate();

    stream.executes(|| {
        let device = Device::default();
        let inputs: Vec<Tensor<2>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &length)| {
                let shape = if dim == 0 {
                    [length, other]
                } else {
                    [other, length]
                };
                Tensor::from_data(
                    TensorData::new(vec![(i + 1) as f32; length * other], shape),
                    &device,
                )
            })
            .collect();
        device.sync().unwrap();

        {
            let inspector = FusionInspector::install(stream);
            let output = Tensor::cat(inputs.clone(), dim).mul_scalar(2.0);
            let data = output.into_data();
            let values = data.as_slice::<f32>().unwrap();
            for (i, &value) in values.iter().enumerate() {
                let coordinate = if dim == 0 { i / other } else { i % total };
                let mut end = 0;
                let source = lengths
                    .iter()
                    .position(|length| {
                        end += length;
                        coordinate < end
                    })
                    .unwrap();
                assert_eq!(value, (source + 1) as f32 * 2.0);
            }
            device.sync().unwrap();
            let reports = inspector.drain();
            assert!(
                reports.iter().flat_map(|r| &r.blocks).any(|block| {
                    matches!(
                        block.kind,
                        BlockKind::Fused {
                            name: "ElementWise",
                            ..
                        }
                    ) && block.operations.iter().any(matchers::is_cat())
                        && block
                            .operations
                            .iter()
                            .any(matchers::is_mul_scalar_float(inputs[0].dtype()))
                }),
                "{pattern:?}, {layout:?}, {count} inputs, target {target_elements}: \
                 concat and multiplication did not fuse: {reports:?}"
            );
        }

        bencher.bench_synced(|| Tensor::cat(inputs.clone(), dim).mul_scalar(2.0));
    });
}

macro_rules! bench_layout {
    ($name:ident, $pattern:expr, $layout:expr) => {
        #[divan::bench_group]
        mod $name {
            use super::*;

            #[divan::bench(args = [2, 5, 16, 32, 64, 128, 256])]
            fn target_1k(bencher: Bencher, count: usize) {
                bench_cat(bencher, count, $pattern, $layout, 1024);
            }

            #[divan::bench(args = [2, 5, 16, 32, 64, 128, 256])]
            fn target_64k(bencher: Bencher, count: usize) {
                bench_cat(bencher, count, $pattern, $layout, 65536);
            }

            #[divan::bench(args = [2, 5, 16, 32, 64, 128, 256])]
            fn target_4m(bencher: Bencher, count: usize) {
                bench_cat(bencher, count, $pattern, $layout, 4194304);
            }
        }
    };
}

macro_rules! bench_pattern {
    ($name:ident, $pattern:expr) => {
        #[divan::bench_group]
        mod $name {
            use super::*;

            bench_layout!(whole_vector, $pattern, Layout::WholeVector);
            bench_layout!(aligned_axis, $pattern, Layout::AlignedAxis);
            bench_layout!(unaligned_axis, $pattern, Layout::UnalignedAxis);
        }
    };
}

bench_pattern!(equal, Pattern::Equal);
bench_pattern!(repeating, Pattern::Repeating);
bench_pattern!(large_first, Pattern::LargeFirst);
bench_pattern!(large_last, Pattern::LargeLast);
