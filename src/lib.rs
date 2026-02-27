//! A library for creating and verifying Groth16 proofs in a collaborative fashion using MPC.
mod groth16_gpu;

mod bridges;
mod gpu_utils;
/// This module contains the Groth16 prover trait
pub mod mpc;
mod utils;
mod verifier;
use icicle_runtime::runtime;

pub use groth16_gpu::{CircomReduction, LibSnarkReduction, R1CSToQAP, Rep3CoGroth16};

pub fn load_backend_from_env_and_set_device(device_idx: i32) {
    runtime::load_backend_from_env_or_default().unwrap();

    // Select CUDA device
    let device = icicle_runtime::Device::new("CUDA", device_idx);
    icicle_runtime::set_device(&device).unwrap();
}

#[cfg(test)]
mod tests {

    use ark_bls12_377::Bls12_377;
    use ark_bn254::Bn254;

    use ark_relations::r1cs::Matrix;
    use ark_serialize::CanonicalDeserialize;
    use circom_types::{CheckElement, Witness, groth16::Zkey};
    use co_circom_types::SharedWitness;
    use co_groth16::{ConstraintMatrices, ProvingKey};
    use itertools::izip;
    use mpc_net::{Network, local::LocalNetwork};

    use std::{fs::File, sync::Arc};

    use crate::{
        CircomReduction, LibSnarkReduction, Rep3CoGroth16, groth16_gpu::Groth16,
        load_backend_from_env_and_set_device,
    };

    macro_rules! run_provers {
        (
        cpu_prove = $cpu_prove:expr,
        gpu_prove = $gpu_prove:expr,
        pkey = $pkey:expr,
        matrices = $matrices:expr,
        witness = $witness:expr,
        silent = $silent:expr
    ) => {{
            use std::time::Instant;

            // ---- CPU prove ----
            if !$silent {
                println!("------------------- Proving (CPU) --------------------\n");
            }
            let t0 = Instant::now();
            let _ = ($cpu_prove)($pkey, $matrices, $witness.clone())
                .expect("CPU proof generation works");
            if !$silent {
                println!("Time taken for CPU proving: {:?}\n", t0.elapsed());

                // ---- GPU warm-up ----
                println!("-------------- Proving (GPU before warm-up) --------------\n");
            }
            let t1 = Instant::now();
            let _ = ($gpu_prove)($pkey, $matrices, $witness.clone())
                .expect("GPU proof generation works (before warm-up)");

            if !$silent {
                println!(
                    "Time taken for GPU proving (before warm-up): {:?}\n",
                    t1.elapsed()
                );

                // ---- GPU final prove ----
                println!("-------------- Proving (GPU after warm-up) --------------\n");
            }
            let t2 = Instant::now();
            let _ = ($gpu_prove)($pkey, $matrices, $witness)
                .expect("GPU proof generation works (after warm-up)");

            if !$silent {
                println!(
                    "Time taken for GPU proving (after warm-up): {:?}\n",
                    t2.elapsed()
                );
            }

            println!("\n");
        }};
    }
    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bn254"]
    fn create_proof_multiplier2_bn254() {
        load_backend_from_env_and_set_device(0);

        for check in [CheckElement::Yes, CheckElement::No] {
            let zkey_file =
                File::open("test_vectors/Groth16/bn254/multiplier2/circuit.zkey").unwrap();
            let witness_file =
                File::open("test_vectors/Groth16/bn254/multiplier2/witness.wtns").unwrap();

            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
            let zkey = Zkey::<Bn254>::from_reader(zkey_file, check).unwrap();
            let (matrices, pkey) = zkey.into();

            let public_input = witness.values[..matrices.num_instance_variables].to_vec();
            let witness = SharedWitness {
                public_inputs: public_input.clone(),
                witness: witness.values[matrices.num_instance_variables..].to_vec(),
            };

            run_provers!(
                cpu_prove =
                    co_groth16::Groth16::<Bn254>::plain_prove::<co_groth16::CircomReduction>,
                gpu_prove = Groth16::<Bn254>::plain_prove::<CircomReduction>,
                pkey = &pkey,
                matrices = &matrices,
                witness = witness,
                silent = false
            );
        }
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bn254"]
    fn create_proof_poseidon_hash_bn254() {
        load_backend_from_env_and_set_device(0);

        for check in [CheckElement::Yes, CheckElement::No] {
            let zkey_file = File::open("test_vectors/Groth16/bn254/poseidon/circuit.zkey").unwrap();
            let witness_file =
                File::open("test_vectors/Groth16/bn254/poseidon/witness.wtns").unwrap();

            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
            let zkey = Zkey::<Bn254>::from_reader(zkey_file, check).unwrap();
            let (matrices, pkey) = zkey.into();
            let public_input = witness.values[..matrices.num_instance_variables].to_vec();
            let witness = SharedWitness {
                public_inputs: public_input.clone(),
                witness: witness.values[matrices.num_instance_variables..].to_vec(),
            };
            run_provers!(
                cpu_prove =
                    co_groth16::Groth16::<Bn254>::plain_prove::<co_groth16::CircomReduction>,
                gpu_prove = Groth16::<Bn254>::plain_prove::<CircomReduction>,
                pkey = &pkey,
                matrices = &matrices,
                witness = witness,
                silent = false
            );
        }
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bn254"]
    fn create_proof_poseidon_hash_bn254_rep3() {
        load_backend_from_env_and_set_device(0);

        for check in [CheckElement::Yes, CheckElement::No] {
            let zkey_file = File::open("test_vectors/Groth16/bn254/poseidon/circuit.zkey").unwrap();
            let witness_file =
                File::open("test_vectors/Groth16/bn254/poseidon/witness.wtns").unwrap();

            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
            let zkey = Zkey::<Bn254>::from_reader(zkey_file, check).unwrap();

            let zkey: (ConstraintMatrices<_>, ProvingKey<_>) = zkey.into();
            let zkey1 = Arc::new(zkey);
            let zkey2 = Arc::clone(&zkey1);
            let zkey3 = Arc::clone(&zkey1);

            let mut rng = rand::thread_rng();
            let [witness_share1, witness_share2, witness_share3] =
                SharedWitness::share_rep3(witness, zkey1.0.num_instance_variables, &mut rng);

            let nets0 = LocalNetwork::new_3_parties();
            let nets1 = LocalNetwork::new_3_parties();

            let mut threads = vec![];
            for (net0, net1, x, zkey) in izip!(
                nets0,
                nets1,
                [witness_share1, witness_share2, witness_share3].into_iter(),
                [zkey1, zkey2, zkey3].into_iter()
            ) {
                threads.push(std::thread::spawn(move || {
                    let cpu_prove = |pkey, matrices, witness| {
                        co_groth16::Rep3CoGroth16::<Bn254>::prove::<
                            LocalNetwork,
                            co_groth16::CircomReduction,
                        >(&net0, &net1, pkey, matrices, witness)
                    };

                    let gpu_prove = |pkey, matrices, witness| {
                        Rep3CoGroth16::<Bn254>::prove::<LocalNetwork, CircomReduction>(
                            &net0, &net1, pkey, matrices, witness,
                        )
                    };

                    if net0.id() == 0 {
                        run_provers!(
                            cpu_prove = cpu_prove,
                            gpu_prove = gpu_prove,
                            pkey = &zkey.1,
                            matrices = &zkey.0,
                            witness = x,
                            silent = false
                        );
                    } else {
                        run_provers!(
                            cpu_prove = cpu_prove,
                            gpu_prove = cpu_prove,
                            pkey = &zkey.1,
                            matrices = &zkey.0,
                            witness = x,
                            silent = true
                        );
                    }
                }));
            }

            let _ = threads.pop().unwrap().join().unwrap();
            let _ = threads.pop().unwrap().join().unwrap();
            let _ = threads.pop().unwrap().join().unwrap();
        }
    }

    fn proof_libsnark_penumbra_bls12_377(circuit: &str) {
        load_backend_from_env_and_set_device(0);

        let pkey_file = File::open(format!(
            "test_vectors/Groth16/bls12_377/{circuit}/circuit.pk"
        ))
        .unwrap();
        let a_file = File::open(format!("test_vectors/Groth16/bls12_377/{circuit}/a.bin")).unwrap();
        let b_file = File::open(format!("test_vectors/Groth16/bls12_377/{circuit}/b.bin")).unwrap();
        let c_file = File::open(format!("test_vectors/Groth16/bls12_377/{circuit}/c.bin")).unwrap();
        let witness_file = File::open(format!(
            "test_vectors/Groth16/bls12_377/{circuit}/witness.wtns"
        ))
        .unwrap();

        let witness = Witness::<ark_bls12_377::Fr>::from_reader(witness_file).unwrap();
        let pkey = ProvingKey::<Bls12_377>::deserialize_uncompressed_unchecked(pkey_file).unwrap();
        // TODO once we can serde ConstraintMatrices, we dont need to do this anymore
        let a = Matrix::<ark_bls12_377::Fr>::deserialize_uncompressed(a_file).unwrap();
        let b = Matrix::<ark_bls12_377::Fr>::deserialize_uncompressed(b_file).unwrap();
        let c = Matrix::<ark_bls12_377::Fr>::deserialize_uncompressed(c_file).unwrap();
        let matrices = ConstraintMatrices {
            num_instance_variables: pkey.b_g1_query.len() - pkey.l_query.len(),
            num_witness_variables: pkey.a_query.len() - pkey.b_g1_query.len() + pkey.l_query.len(),
            num_constraints: a.len(),
            a_num_non_zero: a.len(),
            b_num_non_zero: b.len(),
            c_num_non_zero: c.len(),
            a,
            b,
            c,
        };

        let public_input = witness.values[..matrices.num_instance_variables].to_vec();
        let witness = SharedWitness {
            public_inputs: public_input.clone(),
            witness: witness.values[matrices.num_instance_variables..].to_vec(),
        };
        run_provers!(
            cpu_prove =
                co_groth16::Groth16::<Bls12_377>::plain_prove::<co_groth16::LibSnarkReduction>,
            gpu_prove = Groth16::<Bls12_377>::plain_prove::<LibSnarkReduction>,
            pkey = &pkey,
            matrices = &matrices,
            witness = witness,
            silent = false
        );
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bls12_377"]
    fn proof_libsnark_penumbra_spend_bls12_377() {
        proof_libsnark_penumbra_bls12_377("penumbra_spend");
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bls12_377"]
    fn proof_libsnark_penumbra_output_bls12_377() {
        proof_libsnark_penumbra_bls12_377("penumbra_output");
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bls12_377"]
    fn proof_libsnark_penumbra_delegator_vote_bls12_377() {
        proof_libsnark_penumbra_bls12_377("penumbra_delegator_vote");
    }
}
