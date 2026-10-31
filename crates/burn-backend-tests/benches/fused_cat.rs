//! Measures fused concatenation, including host preparation and device completion.

use burn_fusion::inspect::{BlockKind, FusionInspector, matchers};
use burn_tensor::{Device, StreamId, Tensor, TensorData};
use divan::Bencher;

const CASES: [usize; 252] = {
    let mut cases = [0; 252];
    let mut i = 0;
    while i < cases.len() {
        cases[i] = i;
        i += 1;
    }
    cases
};

fn main() {
    divan::main();
}

/// Inputs and output have rank two. `dim` joins segments; the other axis preserves its length.
#[divan::bench(args = CASES, sample_count = 100)]
fn fused_cat(bencher: Bencher, case: usize) {
    let (count, pattern, layout, size) = if case < 144 {
        (
            [2, 5, 16, 32][case % 4],
            case / 4 % 4,
            case / 16 % 3,
            case / 48,
        )
    } else {
        let extended = case - 144;
        (
            [64, 128, 256][extended % 3],
            extended / 3 % 4,
            extended / 12 % 3,
            extended / 36,
        )
    };
    let target_elements = [1024, 65536, 4194304][size];
    let mut lengths: Vec<usize> = (0..count)
        .map(|i| match pattern {
            0 => 4,
            1 => [1, 2, 5, 6, 3][i % 5],
            2 if i == 0 => count * 8 + 1,
            3 if i == count - 1 => count * 8 + 1,
            _ => 1,
        })
        .collect();
    if layout == 1 {
        for length in &mut lengths {
            *length *= 4;
        }
    } else if layout == 2 {
        // Keep the total aligned while leaving internal boundaries inside vectors.
        let total: usize = lengths.iter().sum();
        lengths[count - 1] += (4 - total % 4) % 4;
    }
    let total: usize = lengths.iter().sum();
    let other = if layout == 0 {
        (target_elements / total / 4).max(1) * 4
    } else {
        (target_elements / total).max(1)
    };
    let dim = usize::from(layout != 0);
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
                "case {case} did not fuse concat and multiplication: {reports:?}"
            );
        }

        for _ in 0..5 {
            let _output = Tensor::cat(inputs.clone(), dim).mul_scalar(2.0);
            device.sync().unwrap();
        }
        bencher.bench(|| {
            let output = Tensor::cat(inputs.clone(), dim).mul_scalar(2.0);
            device.sync().unwrap();
            output
        });
    });
}
