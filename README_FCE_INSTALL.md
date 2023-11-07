### New install instructions
conda create -n ddcolor python=3.8
conda activate ddcolor

~/anaconda3/envs/ddcolor/bin/pip install -r requirements.txt

~/anaconda3/envs/ddcolor/bin/python3 setup.py develop

~/anaconda3/envs/ddcolor/bin/pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115

~/anaconda3/envs/ddcolor/bin/pip freeze >> working_revision.txt


### Launch the broadcaster

```bash
just dmsbroadcaster

./build/dmsbroadcaster/dmsbroadcaster --quartersized
```