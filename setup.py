from setuptools import setup

setup(name='py_portada_paragraphs',
    version='0.0.2',
    description='Process to get paragraphs from documment images using YOLO model in PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    url="https://github.com/portada-git/py_portada_paragraphs",
    packages=['py_portada_paragraphs'],
    py_modules=['py_yolo_paragraphs', 'layout_structure'],
    install_requires=[
	'opencv-python',
	'doclayout-YOLO @ git+https://github.com/opendatalab/DocLayout-YOLO.git',
	'numpy',
    'huggingface_hub',
    ],
    python_requires='>=3.9',
    zip_safe=False)
