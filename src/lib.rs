//! A library for creating and verifying Groth16 proofs in a collaborative fashion using MPC.
mod groth16_gpu;

mod bridges;
mod gpu_utils;
/// This module contains the Groth16 prover trait
pub mod mpc;
mod utils;
mod verifier;

pub use groth16_gpu::{CircomReduction, LibSnarkReduction, R1CSToQAP, Rep3CoGroth16};

#[cfg(test)]
mod tests {

    use ark_bls12_377::Bls12_377;
    use ark_bn254::Bn254;

    use ark_relations::r1cs::Matrix;
    use ark_serialize::CanonicalDeserialize;
    use circom_types::{
        CheckElement, Witness,
        groth16::{Proof as CircomGroth16Proof, VerificationKey as JsonVerificationKey, Zkey},
    };
    use co_circom_types::SharedWitness;
    use co_groth16::{ConstraintMatrices, ProvingKey, VerifyingKey};
    use icicle_runtime::runtime;
    use itertools::izip;
    use rand::thread_rng;

    use mpc_net::local::LocalNetwork;
    use std::{fs::File, sync::Arc};

    use crate::{
        CircomReduction, Rep3CoGroth16,
        groth16_gpu::{Groth16, LibSnarkReduction},
    };

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bn254"]
    fn create_proof_and_verify_bn254() {
        // TODO CESAR: Handle properly
        runtime::load_backend_from_env_or_default().unwrap();

        // Select CUDA device
        let device = icicle_runtime::Device::new("CUDA", 0);
        icicle_runtime::set_device(&device).unwrap();

        for check in [CheckElement::Yes, CheckElement::No] {
            let zkey_file =
                File::open("test_vectors/Groth16/bn254/multiplier2/circuit.zkey").unwrap();
            let witness_file =
                File::open("test_vectors/Groth16/bn254/multiplier2/witness.wtns").unwrap();
            let vk_file =
                File::open("test_vectors/Groth16/bn254/multiplier2/verification_key.json").unwrap();

            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
            let zkey = Zkey::<Bn254>::from_reader(zkey_file, check).unwrap();
            let (matrices, pkey) = zkey.into();
            let vk: JsonVerificationKey<Bn254> = serde_json::from_reader(vk_file).unwrap();
            let vk = vk.into();
            let public_input = witness.values[..matrices.num_instance_variables].to_vec();
            let witness = SharedWitness {
                public_inputs: public_input.clone(),
                witness: witness.values[matrices.num_instance_variables..].to_vec(),
            };
            println!("-------------- Proving (before warm-up) --------------");
            println!("\n");
            let _ =
                Groth16::<Bn254>::plain_prove::<CircomReduction>(&pkey, &matrices, witness.clone())
                    .expect("proof generation works");
            println!("\n");
            println!("-------------- Proving (after warm-up) --------------");
            println!("\n");
            let proof = Groth16::<Bn254>::plain_prove::<CircomReduction>(&pkey, &matrices, witness)
                .expect("proof generation works");
            println!("\n");

            let proof = CircomGroth16Proof::from(proof);
            let ser_proof = serde_json::to_string(&proof).unwrap();
            let der_proof = serde_json::from_str::<CircomGroth16Proof<Bn254>>(&ser_proof).unwrap();
            let der_proof = der_proof.into();
            Groth16::verify(&vk, &der_proof, &public_input[1..]).expect("can verify");
        }
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bn254"]
    fn create_proof_and_verify_poseidon_hash_bn254() {
        // TODO CESAR: Handle properly
        runtime::load_backend_from_env_or_default().unwrap();

        // Select CUDA device
        let device = icicle_runtime::Device::new("CUDA", 0);
        icicle_runtime::set_device(&device).unwrap();

        for check in [CheckElement::Yes, CheckElement::No] {
            let zkey_file = File::open("test_vectors/Groth16/bn254/poseidon/circuit.zkey").unwrap();
            let witness_file =
                File::open("test_vectors/Groth16/bn254/poseidon/witness.wtns").unwrap();
            let vk_file =
                File::open("test_vectors/Groth16/bn254/poseidon/verification_key.json").unwrap();

            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
            let zkey = Zkey::<Bn254>::from_reader(zkey_file, check).unwrap();
            let (matrices, pkey) = zkey.into();
            let vk: JsonVerificationKey<Bn254> = serde_json::from_reader(vk_file).unwrap();
            let vk = vk.into();
            let public_input = witness.values[..matrices.num_instance_variables].to_vec();
            let witness = SharedWitness {
                public_inputs: public_input.clone(),
                witness: witness.values[matrices.num_instance_variables..].to_vec(),
            };
            println!("-------------- Proving (before warm-up) --------------");
            println!("\n");
            let _ =
                Groth16::<Bn254>::plain_prove::<CircomReduction>(&pkey, &matrices, witness.clone())
                    .expect("proof generation works");
            println!("\n");
            println!("-------------- Proving (after warm-up) --------------");
            println!("\n");
            let proof = Groth16::<Bn254>::plain_prove::<CircomReduction>(&pkey, &matrices, witness)
                .expect("proof generation works");
            println!("\n");

            let proof = CircomGroth16Proof::from(proof);
            let ser_proof = serde_json::to_string(&proof).unwrap();
            let der_proof = serde_json::from_str::<CircomGroth16Proof<Bn254>>(&ser_proof).unwrap();
            let der_proof = der_proof.into();
            Groth16::verify(&vk, &der_proof, &public_input[1..]).expect("can verify");
        }
    }

    #[test]
    #[ignore = "Requires building the icicle backend with -DCURVE=bn254"]
    fn create_proof_and_verify_poseidon_hash_bn254_rep3() {
        // TODO CESAR: Handle properly
        runtime::load_backend_from_env_or_default().unwrap();

        // Select CUDA device
        let device = icicle_runtime::Device::new("CUDA", 0);
        icicle_runtime::set_device(&device).unwrap();

        for check in [CheckElement::Yes, CheckElement::No] {
            let zkey_file = File::open("test_vectors/Groth16/bn254/poseidon/circuit.zkey").unwrap();
            let witness_file =
                File::open("test_vectors/Groth16/bn254/poseidon/witness.wtns").unwrap();
            let vk_file =
                File::open("test_vectors/Groth16/bn254/poseidon/verification_key.json").unwrap();

            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
            let zkey = Zkey::<Bn254>::from_reader(zkey_file, check).unwrap();

            let zkey: (ConstraintMatrices<_>, ProvingKey<_>) = zkey.into();
            let zkey1 = Arc::new(zkey);
            let zkey2 = Arc::clone(&zkey1);
            let zkey3 = Arc::clone(&zkey1);
            let public_input = witness.values[..zkey1.0.num_instance_variables].to_vec();

            let vk: JsonVerificationKey<Bn254> = serde_json::from_reader(vk_file).unwrap();
            let vk = vk.into();

            let mut rng = thread_rng();
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
                    Rep3CoGroth16::<Bn254>::prove::<_, CircomReduction>(
                        &net0, &net1, &zkey.1, &zkey.0, x,
                    )
                    .unwrap()
                }));
            }

            let result3 = threads.pop().unwrap().join().unwrap();
            let result2 = threads.pop().unwrap().join().unwrap();
            let result1 = threads.pop().unwrap().join().unwrap();
            assert_eq!(result1, result2);
            assert_eq!(result2, result3);
            let proof = CircomGroth16Proof::from(result1);
            let ser_proof = serde_json::to_string(&proof).unwrap();
            let der_proof = serde_json::from_str::<CircomGroth16Proof<Bn254>>(&ser_proof).unwrap();
            let der_proof = der_proof.into();
            Groth16::verify(&vk, &der_proof, &public_input[1..]).expect("can verify");
        }
    }

    fn proof_libsnark_penumbra_bls12_377(circuit: &str) {
        // TODO CESAR: Handle properly
        runtime::load_backend_from_env_or_default().unwrap();

        // Select CUDA device
        let device = icicle_runtime::Device::new("CUDA", 0);
        icicle_runtime::set_device(&device).unwrap();

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
        let vk_file = File::open(format!(
            "test_vectors/Groth16/bls12_377/{circuit}/circuit.vk"
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

        let vk = VerifyingKey::<Bls12_377>::deserialize_uncompressed_unchecked(vk_file).unwrap();
        let public_input = witness.values[..matrices.num_instance_variables].to_vec();
        let witness = SharedWitness {
            public_inputs: public_input.clone(),
            witness: witness.values[matrices.num_instance_variables..].to_vec(),
        };

        println!("-------------- Proving (before warm-up) --------------");
        println!("\n");
        let _ = Groth16::<Bls12_377>::plain_prove::<LibSnarkReduction>(
            &pkey,
            &matrices,
            witness.clone(),
        )
        .expect("proof generation works");

        println!("\n");
        println!("-------------- Proving (after warm-up) --------------");
        println!("\n");
        let proof =
            Groth16::<Bls12_377>::plain_prove::<LibSnarkReduction>(&pkey, &matrices, witness)
                .expect("proof generation works");
        println!("\n");
        Groth16::<Bls12_377>::verify(&vk, &proof, &public_input[1..]).expect("can verify");
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
