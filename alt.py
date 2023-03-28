from main import tokenize_and_train2
INPUT_FILE = "/mnt/nvme1n1p2/Code/fimficOmega_Trimmed.txt"
#INPUT_FILE = "/mnt/nvme1n1p2/Code/fimfic_Omega_demo.txt"

tokenize_and_train2(INPUT_FILE,
    max_seq_length=512,
    micro_batch_size=8,
    gradient_accumulation_steps=1,
    epochs=1,
    learning_rate=3e-4,
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.01,
    model_name="fimficOmegaDemo")
