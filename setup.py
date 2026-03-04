from setuptools import find_packages, setup
from glob import glob

package_name = 'hearo_speech_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hearo_speech_manager']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/srv', glob('srv/*.srv')),
    ],
    install_requires=[
        'setuptools',
        'pymysql>=1.0.0',
        'openai>=1.0.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'librosa>=0.10.0',
        'webrtcvad>=2.0.10',
        'numpy>=1.20.0,<2.0.0',
        'pydub>=0.25.1',
        'google-cloud-speech>=2.0.0',
    ],
    zip_safe=True,
    maintainer='hjeon',
    maintainer_email='feelgood88@kist.re.kr',
    description='HeaRo Speech Manager - VAD, STT, TTS 통합 패키지',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vad = hearo_speech_manager.nodes.vad:main',
            'stt_node = hearo_speech_manager.nodes.stt_node:main',
            'google_stt = hearo_speech_manager.nodes.google_stt:main',
            'tts_handler = hearo_speech_manager.nodes.HeaRo_TTS_Handler_Node:main',
        ],
    },
)
