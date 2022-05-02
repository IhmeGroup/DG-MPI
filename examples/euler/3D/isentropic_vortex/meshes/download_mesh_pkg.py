import gdown

url = 'https://drive.google.com/file/d/1dZQkIwWiRJhZuiR3bpD8lJOyVOX9EFQb/view?usp=sharing'
output = 'mesh_package_cartesian.tar.gz'
gdown.download(url, output, quiet=False, fuzzy=True)
