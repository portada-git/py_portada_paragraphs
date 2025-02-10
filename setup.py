from setuptools import setup

setup(name='py_portada_paragraphs',
    version='0.0.6',
    description='Process to get paragraphs from documment images using YOLO model in PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    url="https://github.com/portada-git/py_portada_paragraphs",
    packages=['py_portada_paragraphs'],
    py_modules=[
	'py_yolo_paragraphs', 
	'layout_structure', 
	'portada_cut_in_paragraphs', 
	'py_portada_utility_for_layout',
	'py_yolo_layout'
    ],
    install_requires=[
	'opencv-python >= 4.8,<4.9',
	'doclayout-YOLO @ git+https://github.com/opendatalab/DocLayout-YOLO.git',
	'numpy < 2',
    	'huggingface_hub',
	'urllib3',
	'ultralytics'
    ],
    python_requires='>=3.9',
    zip_safe=False)
