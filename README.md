# cog-van-gogh
Run these commands in the terminal. It'll clone this repo and download the van-gogh-diffusion model from huggingface:
```bash
git clone https://github.com/satani99/cog-van-gogh.git
cd cog-von-gogh
git lfs install
git clone https://huggingface.co/dallinmackay/Van-Gogh-diffusion
```

After that run below commands to run model using cog:
```bash
cog predict -i image=@input.jpeg
```
To push model on replicate:
```bash
cog login
cog push r8.im/<your-username>/<your-model-name>
```
