#The execution code of specific datasets under different backbones

circo-test-base-openai
python main.py --dataset circo --split test  --dataset-path data/CIRCO --llm_prompt prompts.structural_modifier_prompt --clip ViT-B/32

circo-test-large-openai
python main.py --dataset circo --split test  --dataset-path data/CIRCO --llm_prompt prompts.structural_modifier_prompt --clip ViT-L/14

circo-test-giga
python main.py --dataset circo --split test  --dataset-path data/CIRCO --llm_prompt prompts.structural_modifier_prompt --clip ViT-bigG-14


CIRR-test-base-openai
python main.py --dataset cirr --split test  --dataset-path data/CIRR --llm_prompt prompts.contextual_modifier_prompt --clip ViT-B/32

CIRR-test-large-openai
python main.py --dataset cirr --split test  --dataset-path data/CIRR --llm_prompt prompts.contextual_modifier_prompt --clip ViT-L/14

CIRR-test-giga
python main.py --dataset cirr --split test  --dataset-path data/CIRR --llm_prompt prompts.contextual_modifier_prompt --clip ViT-bigG-14



fashioniq-shirt-base-openai
python main.py --dataset fashioniq_shirt --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B/32

fashioniq-shirt-large-openai
python main.py --dataset fashioniq_shirt --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L/14

fashioniq-shirt-giga
python main.py --dataset fashioniq_shirt --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-bigG-14


fashioniq-dress-base-openai
python main.py --dataset fashioniq_dress --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B/32

fashioniq-dress-large-openai
python main.py --dataset fashioniq_dress --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L/14

fashioniq-dress-giga
python main.py --dataset fashioniq_dress --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-bigG-14


fashioniq-toptee-base-openai
python main.py --dataset fashioniq_toptee --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B/32

fashioniq-toptee-large-openai
python main.py --dataset fashioniq_toptee --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-L/14

fashioniq-toptee-giga
python main.py --dataset fashioniq_toptee --split val  --dataset-path data/FASHIONIQ --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-bigG-14
