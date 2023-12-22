import logging
import os
import muspy
import music21
from music21 import *
import numpy as np
import pypianoroll
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config.train_config import TrainingConfig

path = './dataset/bachdatas'

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def sliding_inds(n, seq_length, step_length):
    return np.arange(n-seq_length, step=step_length)

def sliding_window(roll, seq_length, step_length=1):
    """
    returns [n x seq_length x 88]
        if step_length == 1, then roll[i,1:] == roll[i+1,:-1]
    """
    rolls = []
    if roll.shape[0] <= seq_length:
        temp = np.zeros((seq_length, roll.shape[1]), dtype=roll.dtype)
        temp[:roll.shape[0]] = roll
        return np.array([temp])
    for i in sliding_inds(roll.shape[0], seq_length, step_length):
        rolls.append(roll[i:i+seq_length,:])
    if len(rolls) == 0:
        raise("value is None")
    return np.dstack(rolls).swapaxes(0,2).swapaxes(1,2)

def songs_to_pianoroll(pianosongs, seq_length=256, step_length=1):
    """
    songs = [song1, song2, ...]
    """
    rolls = [sliding_window(s.pianoroll, seq_length, step_length) for s in pianosongs]
    rolls = [r for r in rolls if len(r) > 0]  # 将空集去除
    inds = [i*np.ones((len(r),)) for i,r in enumerate(rolls)]
    # return np.vstack(rolls), np.hstack(inds)
    return np.array(rolls)

def split_pianoroll(roll, seq_length=256, step_length=1):
    rolls = []
    if roll.shape[0] <= seq_length:
        temp = np.zeros((1, seq_length, roll.shape[1]))
        temp[0, :roll.shape[0]] += roll
        return temp
    for i in sliding_inds(roll.shape[0], seq_length, step_length):
        rolls.append(roll[i:i+seq_length,:])
    if len(rolls) == 0:
        return np.array([])
    return np.array(rolls)

def load(path):
    """加载bach数据并保存到path路径"""
    bach_dataset = music21.corpus.chorales.Iterator()
    for i, music in tqdm(enumerate(bach_dataset)):
        mf = music21.midi.translate.music21ObjectToMidiFile(music)
        output_file = os.path.join(path, f'bach{i}.mid')
        mf.open(output_file, 'wb')
        mf.write()
        mf.close()

def midi_to_pianoroll(path, seq_length=256, step_length=TrainingConfig.resolution):
    """return: [batch, voices, seq_len, 128]
        default: (27271, 4, 256, 128)
        (11192, 4, 128, 128)
    """
    split_track = []
    # 判断文件夹是否为空
    namelist = os.listdir(path)
    if len(namelist) == 0:
        load(path)
    
    for name in tqdm(namelist):
        filepath = os.path.join(path, name)
        multitrack = pypianoroll.read(filepath, collect_onsets_only=True, resolution=TrainingConfig.resolution)  # 保留节奏信息
        if len(multitrack.tracks) > 4:
            continue
        splitsongs = songs_to_pianoroll(multitrack.tracks, seq_length, step_length)
        swapped_voice = splitsongs.swapaxes(0, 1)  # [voices, splits, seq_len, 128]
        split_track.append(swapped_voice)
    data = np.vstack(split_track)
    np.save('./dataset/bach_pianoroll_128_4_1', data)
    print(f'bach pianoroll data saved. datashape: {data.shape}')
    return np.vstack(split_track)

def midi_to_pianoroll_blend(seq_length=128, step_length=4):
    """return: [batch, seq_len, 128]
        default:256,  (27701, 256, 128)
    """
    tracks = []
    bach_dataset = music21.corpus.chorales.Iterator()
    for i, data in enumerate(bach_dataset):
        parts = data.getElementsByClass(stream.Part)
        if len(parts) != 4:
            continue
        music = muspy.inputs.from_music21(data, resolution=TrainingConfig.resolution)
        music = muspy.to_pianoroll_representation(music, encode_velocity=False)
        track = split_pianoroll(music, seq_length, step_length)
        tracks.append(track)
    
    data = np.vstack(tracks)
    np.save('./dataset/bach_pianoroll_blend_128', data)
    print(f'bach pianoroll data saved. datashape: {data.shape}')
    return np.vstack(track)

def compute_voice_ranges():
    """[36, 88]"""
    voice_ranges = [129, -1]
    print('Computing voice ranges')
    bach_dataset = music21.corpus.chorales.Iterator()
    for voice in tqdm(bach_dataset):
        midi_pitches = []
        for note in voice.flat.notes():
            if 'Note' in note.classes:
                midi_pitches.append(note.pitch.midi)
        min_midi, max_midi = min(midi_pitches), max(midi_pitches)
        if min_midi < voice_ranges[0]:
            voice_ranges[0] = min_midi
        if max_midi > voice_ranges[1]:
            voice_ranges[1] = max_midi
    print(voice_ranges)
    return voice_ranges

def savemusic(image_array, test_dir, epoch):
    """按照epoch对矩阵进行存储为mid类型文件"""
    for i in range(image_array.shape[0]):
        my_music = muspy.from_pianoroll_representation(image_array[i,:, :, 0], resolution=TrainingConfig.resolution, encode_velocity=False, default_velocity=90)

        filename1 = f'MIDI_{epoch}_{i}.mid'
        path = os.path.join(test_dir, filename1)
        muspy.write_midi(path , my_music)

def trans_bool_2d(image):
    """输入tensor类型数据，转换为bool类型的array数据"""
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().numpy()
    image = image > 0.5
    return image

def trans_bool_3d(image):
    """输入tensor类型数据，转换为bool类型的array数据"""
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = image > 0.5
    return image

class MidiDataset(Dataset):
    """Pre-processed MIDI dataset."""
    def __init__(self, npy_file, midi_start=36, midi_end=96, compress=False, transform=None, blend=False):
        self.piano_rolls = np.load(npy_file)
        if blend:
            "声部是否在同一通道"
            self.piano_rolls_blend = np.zeros_like(self.piano_rolls)[:,0]
            for idx in range(self.piano_rolls.shape[1]):
                self.piano_rolls_blend += self.piano_rolls[:,idx]
            self.piano_rolls = self.piano_rolls_blend
        self.piano_rolls = self.piano_rolls > 0
        self.piano_rolls = np.array(self.piano_rolls, dtype=np.float32)
        self.transform = transform

        # 中心化，将0，1改为-1，1
        # self.piano_rolls = np.where(self.piano_rolls > 0, self.piano_rolls, -1)

        if compress:
            # 限制在古典乐的音域范围内
            self.piano_rolls = self.piano_rolls[:, :, midi_start:midi_end]

    def __len__(self):
        return len(self.piano_rolls)

    def get_mem_usage(self):
        """
            Returns the memory usage in MB
        """
        return self.piano_rolls.memory_usage(deep=True).sum() / 1024**2

    def _get_indexer(self):
        """
            Get an indexer that treats each first level index as a sample.
        """
        return self.piano_rolls.index.get_level_values(0).unique()

    def __getitem__(self, idx):
        piano_roll = self.piano_rolls[idx]
        sample = {"images": piano_roll}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def checkfile(path):
    if not os.path.exists(path):
        os.makedirs(path)