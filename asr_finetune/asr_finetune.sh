# python examples/asr/speech_to_text_finetune.py \
#     --config-path=<path to dir of configs> \
#     --config-name=<name of config without .yaml>) \
#     model.train_ds.manifest_filepath="<path to manifest file>" \
#     model.validation_ds.manifest_filepath="<path to manifest file>" \
#     model.tokenizer.update_tokenizer=<True/False> \ # True to update tokenizer, False to retain existing tokenizer
#     model.tokenizer.dir=<path to tokenizer dir> \ # Path to tokenizer dir when update_tokenizer=True
#     model.tokenizer.type=<tokenizer type> \ # tokenizer type when update_tokenizer=True
#     trainer.devices=-1 \
#     trainer.accelerator='gpu' \
#     trainer.max_epochs=50 \
#     +init_from_nemo_model="<path to .nemo model file>" (or +init_from_pretrained_model="<name of pretrained checkpoint>")

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 python asr_finetune.py \
    --config-path="/tmp2/b10902031/AICUP/asr_finetune" \
    --config-name="speech_to_text_finetune.yaml" \
    model.train_ds.manifest_filepath="/tmp2/b10902031/AICUP/train/asr_finetune_trainset.jsonl" \
    model.validation_ds.manifest_filepath="/tmp2/b10902031/AICUP/valid/asr_finetune_validset.jsonl" \
    model.tokenizer.update_tokenizer=False \
    trainer.devices=-1 \
    trainer.accelerator='gpu' \
    trainer.max_epochs=50 \
    +init_from_nemo_model="/tmp2/b10902031/AICUP/models/nvidia/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo" #(or +init_from_pretrained_model="<name of pretrained checkpoint>")