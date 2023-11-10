import gdown

url = "https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download"
output = "GoogleNews-vectors-negative300.bin.gz"
gdown.download(url, output, quiet=False)
