# Openvla-libero-repro-study

## Project Overview

Hardware specifications and environment dependencies are listed below:

### Hardware

- GPU: NVIDIA GeForce RTX 5090
- GPU memory: 32,607 MiB; compute capability: 12.0 (sm_120)
- GPU power limit: 600 W
- System memory: 31 GiB

### Environment

- Python 3.10
- PyTorch built with CUDA 12.8

### Core Dependencies

- `torch==2.9.1+cu128`
- `torchvision==0.24.1+cu128`
- `torchaudio==2.9.1+cu128`
- `flash-attn==2.8.3+cu128sm120`
- `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`, `peft==0.11.1`, `accelerate==1.14.0`
- `numpy==1.26.4`
- `tensorflow==2.15.0`, `tensorflow-datasets==4.9.3`, `tensorflow-graphics==2021.12.3`, `tensorflow-metadata==1.14.0`
- `protobuf==3.20.3`, `absl-py==1.4.0`
- `dlimp==0.0.1`
- `wandb==0.15.12`, `setuptools==75.8.0`, `huggingface-hub==0.23.5`

### LIBERO Simulation Dependencies

- `robosuite==1.4.1`, `mujoco==3.1.6`, `bddl==3.6.0`
- `LIBERO==0.1.0`

This environment is used to deploy the OpenVLA-7B base model and set up the dependencies for inference, LoRA fine-tuning, and LIBERO evaluation. For setup details and troubleshooting notes, see the [openvla_notes](https://blog.csdn.net/ccccc121212wtf/article/details/164843753).


## Fine-tuning

The fine-tuning configuration is shown below. A checkpoint is saved every 500 training steps. The training data is derived from LIBERO-Spatial (libero_spatial_no_noops).

<div align="center">
  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th></tr>
    </thead>
    <tbody>
      <tr><td>Base model</td><td>OpenVLA-7B</td></tr>
      <tr><td>Dataset</td><td>libero_spatial_no_noops</td></tr>
      <tr><td>Batch size</td><td>4</td></tr>
      <tr><td>Gradient accumulation steps</td><td>4</td></tr>
      <tr><td>Precision</td><td>BF16</td></tr>
      <tr><td>LoRA rank</td><td>32</td></tr>
      <tr><td>LoRA dropout</td><td>0</td></tr>
      <tr><td>Learning rate</td><td>5 × 10⁻⁴</td></tr>
      <tr><td>Training steps</td><td>8,000</td></tr>
      <tr><td>Image augmentation</td><td>Enabled</td></tr>
    </tbody>
  </table>
</div>

 **Train loss:**
<img width="1380" height="854" alt="image" src="https://github.com/user-attachments/assets/9095bcf0-5199-48e8-b49f-6883c71b4ad9" />

Train loss measures the difference between predicted action tokens and their training labels; it is the loss used to update the model. It drops sharply at the beginning and then gradually decreases from around 3 to about 2.2. Despite fluctuations between steps, the overall downward trend suggests that the model fits the training data better over time.


 **L1 loss:**
<img width="1396" height="856" alt="image" src="https://github.com/user-attachments/assets/3820d8d6-a464-42bf-b935-6728cadb8133" />


This metric decodes predicted and target action tokens into continuous action values and computes their mean absolute difference. It gradually falls to roughly 0.07–0.09 toward the end of training, indicating smaller action prediction errors on training batches. Occasional spikes remain, but the overall trend is downward. These values should not be interpreted directly as end-effector errors in centimeters.


 **Action accuracy:** 
<img width="1394" height="864" alt="image" src="https://github.com/user-attachments/assets/38dde750-a699-48f3-83cc-47bc9caddb1e" />


Action accuracy is the fraction of discrete action tokens that exactly match their target tokens; it is not task success rate. It rises quickly early in training, then fluctuates mostly around 0.36–0.40, with a modest overall increase after roughly 6,000 steps. This suggests improved prediction of training labels, although individual measurements remain noisy. 

 **GPU Utilization :**
<img width="1388" height="871" alt="image" src="https://github.com/user-attachments/assets/5f476615-4071-4b70-a6e2-e8fc55d5999a" />


This metric shows the GPU’s overall compute utilization at each sample. Over approximately 200 minutes, utilization fluctuated mostly between 60% and 90%, with occasional brief dips, indicating sustained GPU activity during training.

 **GPU Memory Used :**
<img width="1383" height="846" alt="image" src="https://github.com/user-attachments/assets/cd8abc86-fde9-4b03-a5cb-f387cf39cbf5" />

This metric shows GPU memory in use, rather than memory utilization as a percentage. Usage stayed roughly between 28.85 and 29.25 GiB, with several step changes after about 5,000 training steps. 



## Evaluation Protocol and Results

For a more detailed summary and analysis of the issues encountered, see the [openvla_notes](https://blog.csdn.net/ccccc121212wtf/article/details/164843753).
Checkpoints from steps 6,000–8,000 were selected because the training loss was relatively stable during this period. The LoRA adapter weights from each checkpoint were merged into the base model before evaluation. Each merged model was evaluated on all 10 LIBERO-Spatial tasks, with 50 trials per task (500 trials per checkpoint). The success rates are shown below.

<div align="center">
  <table>
    <thead>
      <tr><th>Checkpoint</th><th>Success Rate</th></tr>
    </thead>
    <tbody>
      <tr><td>6k steps</td><td>8.80%</td></tr>
      <tr><td>6.5k steps</td><td>15.40%</td></tr>
      <tr><td>7k steps</td><td>23.00%</td></tr>
      <tr><td>7.5k steps</td><td>7.80%</td></tr>
      <tr><td>8k steps</td><td>26.60%</td></tr>
    </tbody>
  </table>
</div>

Closed-loop performance did not improve monotonically with training steps. The success rate increased from 6k to 7k, dropped sharply at 7.5k, and recovered at 8k, which achieved the highest overall success rate among the evaluated checkpoints. In some failed episodes, the robot arm appeared stationary. Episode-level action diagnostics showed very small end-effector pose action values and zero recorded change between consecutive actions, making the arm appear to perform no operation.

The success rates for individual tasks at the best-performing 8k checkpoint are shown below.

<div align="center">
  <table>
    <thead>
      <tr><th>Task</th><th>Success Rate</th></tr>
    </thead>
    <tbody>
      <tr><td>Task 1</td><td>26%</td></tr>
      <tr><td>Task 2</td><td>40%</td></tr>
      <tr><td>Task 3</td><td>10%</td></tr>
      <tr><td>Task 4</td><td>74%</td></tr>
      <tr><td>Task 5</td><td>50%</td></tr>
      <tr><td>Task 6</td><td>26%</td></tr>
      <tr><td>Task 7</td><td>4%</td></tr>
      <tr><td>Task 8</td><td>10%</td></tr>
      <tr><td>Task 9</td><td>0%</td></tr>
      <tr><td>Task 10</td><td>26%</td></tr>
    </tbody>
  </table>
</div>

All 50 Task 9 trials failed, and 44 of the videos showed almost no visible movement. Task 5 had no nearly stationary videos, but 25 trials still failed, suggesting that grasping, placement, or fine control were the main difficulties for this task.

### Successful episodes demos：

**Task1 ：pick up the black bowl between the plate and the ramekin and place it on the plate.**


https://github.com/user-attachments/assets/44596320-39e6-41f6-82ca-77c5fca5689c


**Task5 : pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate.**


https://github.com/user-attachments/assets/d0d37a0e-66f2-42a0-9a6c-2bf400e044bf


**Task10 : pick up the black bowl on the wooden cabinet and place it on the plate**  


https://github.com/user-attachments/assets/758e93e4-61f0-48b1-b64e-b45d2cf9cb2d


### Failed Episodes demos:


Among episodes with clear movement that ultimately failed, some failure modes were readily visible on inspection, such as:


**Task 1, episode 1: The arm appeared to pause for a long time before moving, eventually timing out.**


https://github.com/user-attachments/assets/d77109ac-083f-4fce-9e90-7766fc60a87c


**Task 1, episode 2: After several missed grasps, the arm became almost stationary. It also showed pronounced jitter after the first missed grasp.**


https://github.com/user-attachments/assets/430b0e72-1a85-44bf-b5b2-184c77ed5d22


**Task 1, episode 5: The arm grasped the object successfully and moved smoothly, but placed it off target.**


https://github.com/user-attachments/assets/6e915d92-dd23-4daa-be94-9589fff1a6f0


**Task 5, episode 203: The gripper approached at too large an angle and caught on the drawer.**


https://github.com/user-attachments/assets/8c885118-f426-4aa9-8a4f-4013cdc8cbdf


Some episodes appeared successful on visual inspection but were marked as failures by the evaluation program. For example:


**Task 9: Pick up the black bowl next to the plate and place it on the plate.**
**episodes 408 and 418**


https://github.com/user-attachments/assets/6bbd26d2-2577-47e5-8b4f-6b84f4f86658


https://github.com/user-attachments/assets/2f76b20a-f017-46e4-9f83-78765c96ef65


**Task10, episode452   episode459**  

https://github.com/user-attachments/assets/b1b80872-8eee-48ae-9e0d-1aea43d3392b


https://github.com/user-attachments/assets/35438b33-53a8-4d13-9e9c-6ca35b7f8f2e


The rendered scene can make a placement appear successful even when the bowl rests on the plate’s rim, sits slightly off-center, remains suspended by the gripper, or appears to be inside the plate without the required collision contact. LIBERO, however, determines success using strict geometric and contact conditions.

During evaluation, `done` is checked after every action, and an episode ends immediately when success is detected. Episodes 408, 418, and 452 reached the timeout, so the success condition was not met at the end of any checked control step. The evaluation script did not overlook a state it had already marked as successful.

## Acknowledgements

This project builds on [OpenVLA](https://github.com/openvla/openvla) and uses the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO). The write-up for this project was also informed by [this reproduction project](https://github.com/Escapist-coder/OpenVLA-Libero-Reproduction-Finetune).
