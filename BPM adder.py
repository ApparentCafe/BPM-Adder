#-*- coding: utf8
#!/usr/bin/env python3.7

# Get FFMPEG files https://www.ffmpeg.org/download.html

# HOW TO USE THIS PROGRAM
# Either type in the folder with the songs into the SONG_FOLDER variable
# or pass the folder as the first argument in a command line such as
# python.exe "folder/to/BPM adder.py" C:\folder to the song directory\
# Quotes around the last pathname are optional
#
# Press the spacebar (while in any window) on the beat, and it will automatically stop when it has enough beat information
# It will write the BPM to the song file and move onto the next file
#
# Press escape to indicate the song does not have a BPM and it will write "Unknown" as the BPM
#
# This program will go through all songs without a BPM, then it will go through all songs without a BPM written by this program
# This program writes a marker under the often unused tag 'musicbrainz_discid' to mark that this program has added the BPM
#
# ONLY WORKS ON MP3 FORMATTED FILES!

# SETTINGS
FFMPEG_FOLDER = r"C:\folder to your downloaded FFMPEG.exe files"
SONG_FOLDER = r"C:\folder to your songs that you want to process"
MIN_SONG_LENGTH = 50 # Measured in seconds
MAX_SONG_LENGTH = 60 * 10

import os
os.chdir(FFMPEG_FOLDER)

import time
import threading
import sys
from pathlib import Path
from mutagen.easyid3 import EasyID3
import mutagen
from pydub import AudioSegment
import simpleaudio
import queue
import re
from pynput.keyboard import Key, Listener
import numpy as np
from scipy.stats import zscore

songpath = Path(SONG_FOLDER)
SELFID = 'bpmadder v1.0'
ID3_SELFID_TAG = 'musicbrainz_discid'

filequeue = queue.Queue(10)
spacebarqueue = queue.Queue(0)
ENABLE_THREADING = True
CUTOFF_SCORE = 1e-9

def calc_bpm(arr, v2 = None):
    # Input: Either a list of 2 times, or two separate time values
    # Returns a simple bpm calculation based on two time values in seconds.
    # Returns None if not enough values to use for the bpm
    if v2 is not None:
        v1 = arr
    else:
        try:
            if len(arr) == 2:
                v1, v2 = arr[-2:]
            elif len(arr) < 2:
                return None
            else:
                raise TypeError("An array of length 2 was expected to calculate the bpm")
        except TypeError:
            raise TypeError("An array of length 2 was expected to calculate the bpm")
    return np.inf if (v2 - v1) == 0 else max(int(60 / (v2 - v1)), 1e-8)

def calc_avg_bpm(arr):
    if len(arr) < 2:
        return None
    return 60/np.mean(np.diff(arr))

def percent_overlap(r1, r2):
    # Input: Two time ranges in the form of (start1, end1), (start2, end2)
    # Will calculate whether one range is entirely encompassed within the other, returning 1
    # or whether they are disjoint, returning 0
    # or whether they overlap, in which case (overlap length) / (shortest range) is returned

    a, b = min(r1), max(r1)
    c, d = min(r2), max(r2)

    # Find the overlap between the two ranges
    overlap_start = max(a, c)
    overlap_end = min(b, d)

    # Calculate the length of the overlap
    overlap = overlap_end - overlap_start

    # Calculate the length of each range
    l1 = b - a
    l2 = d - c

    # Calculate the percent overlap based on the average length of the two ranges
    if min(l1, l2) == overlap:
        percent = 1
    elif overlap < 0:
        percent = 0
    else:
        percent = overlap / min(l1, l2)

    return percent

def write_bpm_to_file(soundfile, bpm):
    id3 = soundfile.id3
    filepath = Path(soundfile.filepath)
    if not filepath.exists():
        print(f"ERROR: File path does not exist {filepath}")
        return

    try:
        with filepath.open('a'):
            pass
    except PermissionError:
        print(f"FAILED: Write protected file {filepath}")
        return

    if 'bpm' in id3 and (re.match('^\d+$', id3['bpm'][0]) == None or type(bpm) != type(0) or abs(int('0' + id3['bpm'][0]) - bpm) > 5):
        print(f"Big change in BPM from {id3['bpm'][0]} to {bpm}")
    id3['bpm'] = [str(bpm)]
    id3[ID3_SELFID_TAG] = SELFID
    try:
        id3.save()
    except mutagen.MutagenError:
        print(f"FAILED to write id3 data {filepath}")

def has_bpm(filepath):
    try:
        id3 = EasyID3(filepath)
        if 'bpm' in id3 and len(id3['bpm'][0]):
            return True
    except mutagen.id3._util.ID3NoHeaderError:
        pass
    except mutagen.id3._util.error:
        pass
    return False

def has_BPMAdder_bpm(filepath):
    try:
        id3 = EasyID3(filepath)
        if 'bpm' in id3 and len(id3['bpm'][0]):
            if ID3_SELFID_TAG in id3 and id3[ID3_SELFID_TAG][0] == SELFID:
                return True
    except mutagen.id3._util.ID3NoHeaderError:
        pass
    except mutagen.id3._util.error:
        pass
    return False

def play_sample(s):
    return simpleaudio.play_buffer(s.get_array_of_samples(), s.channels, s.frame_width // s.channels, s.frame_rate)

def estimate_bpm_simple(beat_times):
    # Input: An array of times when the beat occurred in seconds
    # Returns an average estimate of the BPM and a score as to whether it is accurate enough to be used
    if len(beat_times) < 2:
        return None, None, None
    # Truncate data to past 40 beats or 30 seconds
    beat_times = beat_times[-40:]
    beat_times = [x for x in beat_times if x > beat_times[-1] - 30]
    time_diffs = np.diff(beat_times)
    # Find outliers using z-score
    outliers = np.abs(zscore(time_diffs)) > 2
    strong_outliers = np.abs(zscore(time_diffs)) > 2.5
    # Remove outliers from time differences
    time_diffs_no_outliers = time_diffs[~outliers]
    bpm = 60 / np.median(time_diffs_no_outliers)
    # Calculate a score for deciding when we have a good estimate
    sample_mean_std = np.std(time_diffs[~strong_outliers]) / len(time_diffs[~strong_outliers])**0.5
    TOLERANCE = max(3/180, 1.5/bpm)
    return round(bpm), 100 if len(time_diffs_no_outliers) > 3 and sample_mean_std < TOLERANCE else 0, {"diffs":time_diffs_no_outliers, 'outliers':time_diffs[outliers]}

def estimate_bpm_complex(beat_times):
    # Input: An array of times when the beat occurred in seconds
    # Returns an average estimate of the BPM and a score as to whether it is accurate enough to be used
    SECTION_VARIATION = 0.20
    SIMILARITY_CUTOFF = 5 / 180
    FINAL_PERCENT_CUTOFF = 2.5 / 180
    FRONT_CUTOFF = 0.5
    PERCENT_INTERQUARTILE_OVERLAP = 0.50

    # Remove any keypresses made in the first FRONT_CUTOFF seconds and only uses the 50 most recent data points
    beat_times = [v for v in beat_times if v > FRONT_CUTOFF][-50:]

    sections = []
    if len(beat_times) <= 1:
        return None, None, None

    # Slice the beat_times into different sections, where each section has a similar time between each beat.
    curr_section = []

    for i in range(len(beat_times)):
        if len(curr_section) <= 1:
            curr_section.append(beat_times[i])
        else:
            avg_bpm = calc_avg_bpm(curr_section)
            c1 = abs(calc_bpm(beat_times[i - 1:i + 1]) / avg_bpm - 1) <= SECTION_VARIATION
            # If 3 consecutive beat times average to within our current bpm, accept them
            c2 = abs(calc_avg_bpm(beat_times[i - 1:i + 2]) / avg_bpm - 1) <= SECTION_VARIATION / 2
            c3 = abs(calc_avg_bpm(beat_times[i - 2:i + 1]) / avg_bpm - 1) <= SECTION_VARIATION / 2
            if c1 or c2 or c3:
                if c1:
                    new_var = abs(calc_bpm(beat_times[i - 1:i + 1]) / calc_bpm(curr_section[-2:]) - 1)
                elif c2:
                    new_var = abs(np.mean(60 / np.diff(beat_times[i - 1:i + 2])) / calc_bpm(curr_section[-2:]) - 1)
                else:
                    new_var = abs(np.mean(60 / np.diff(beat_times[i - 2:i + 1])) / calc_bpm(curr_section[-2:]) - 1)

                curr_section.append(beat_times[i])
            else:
                sections.append({'bpm': calc_avg_bpm(curr_section), 'sec': curr_section})
                curr_section = [beat_times[i]]

    if len(curr_section) > 1:
        sections.append({'bpm': calc_avg_bpm(curr_section), 'sec': curr_section})

    # Shift the boundaries of each section. Some beat times may be better in a neighbouring group, so move them there.
    if len(sections) > 1:
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(sections) - 1:
                # Check if we should move beat forward to next section
                if abs(calc_bpm(sections[i]['sec'][-2:]) - sections[i]['bpm']) > abs(calc_bpm(sections[i + 1]['sec'][-2:]) - sections[i]['bpm']):
                    sections[i]['sec'] = sections[i]['sec'][:-1]
                    sections[i + 1]['sec'] = sections[i]['sec'][-1:] + sections[i + 1]['sec']
                    changed = True
                if len(sections[i]['sec']) < 2:
                    sections.pop(i)
                else:
                    i += 1

    # Calculate some helpful statistics on each section
    for v in sections:
        v['bpm'] = calc_avg_bpm(v['sec'])
        v['est_beat_time'] = np.mean(np.diff(v['sec']))
        v['sample_count'] = len(v['sec']) - 1
        # Calculate how offset in seconds the data is from an idealized array of beat times with 0 variation
        v['offset'] = abs(np.mean(np.array(v['sec']) - v['sec'][0] - np.array(range(len(v['sec']))) * v['est_beat_time']))
        beat_times_hat = np.array(range(len(v['sec']))) * v['est_beat_time'] + v['offset']
        v['std'] = np.std(np.array(v['sec']) - v['sec'][0] - beat_times_hat)

    # Group all the similar sections together based on a similarity cutoff in order to combine their data
    to_sort = [(sections[i]['est_beat_time'], i) for i in range(len(sections))]
    to_sort.sort()
    groups = [[]]
    group_length = [0]
    group_count = [0]
    for i, v in enumerate(to_sort):
        if len(groups[-1]) == 0 or v[0] / to_sort[i - 1][0] - 1 < SIMILARITY_CUTOFF or abs(60 / v[0] - 60 / to_sort[i - 1][0]) < 1.5 or percent_overlap(np.percentile(np.diff(sections[v[1]]['sec']), [25, 75]), np.percentile(np.diff(sections[groups[-1][-1]]['sec']), [25, 75])) > PERCENT_INTERQUARTILE_OVERLAP:
            groups[-1].append(v[1])
            group_count[-1] += sections[v[1]]['sample_count']
            group_length[-1] += sections[v[1]]['sec'][-1] - sections[v[1]]['sec'][0]
        else:
            groups.append([v[1]])
            group_count.append(0)
            group_length.append(0)
            group_count[-1] += sections[v[1]]['sample_count']
            group_length[-1] += sections[v[1]]['sec'][-1] - sections[v[1]]['sec'][0]
    max_ind = np.argmax(group_count)

    final_bpm = np.average([sections[i]['bpm'] for i in groups[max_ind]],weights=[sections[i]['sample_count'] for i in groups[max_ind]])
    final_std = np.average([sections[i]['std'] for i in groups[max_ind]],weights=[sections[i]['sample_count'] for i in groups[max_ind]])
    final_sample_std = final_std / group_count[max_ind] ** 0.5

    # Have a minimun of 5 seconds or 5 beats of data
    criteria1 = group_length[max_ind] > 5 and group_count[max_ind] > 5
    # Have best group be more than 60% of the data
    criteria2 = group_length[max_ind] > (beat_times[-1] - beat_times[0]) * 0.6 or group_count[max_ind] > len(beat_times) * 0.6
    # Have the error range be less than an acceptable range for BPMs or have at least 9 to 15 beats in the best group
    std_cutoff = abs(60 / final_bpm - 60 / (final_bpm * (1 + FINAL_PERCENT_CUTOFF)))
    criteria3 = final_sample_std * 2 < max(std_cutoff, 1.5 / final_bpm) or group_count[max_ind] >= max(9, 8 * 60 / 125 / (60 / (max(0, final_bpm - 125) ** 1.1 + 125)))
    score = 100 if criteria1 and criteria2 and criteria3 else 0

    return final_bpm, score, [max_ind, groups, sections, [criteria1, criteria2, criteria3]]

def song_player(io):
    global filequeue, spacequeue
    SAMPLE_DURATION = 60
    SAMPLE_LOCATION = 2 / 5

    file_count = 0
    sfile = None
    try:
        while not io.shouldstop:
            try:
                if sfile is not None:
                    sfile.unloadSound()
                sfile = filequeue.get_nowait()
                file_count += 1
            except queue.Empty:
                break # Reached end of queue. This thread can always be restarted if needed.

            file_duration = sfile.sound.duration_seconds
            if file_duration > MIN_SONG_LENGTH and file_duration < MAX_SONG_LENGTH:
                start_tm = time.time()
                sampledur = SAMPLE_DURATION
                samplestart = max(0, min(file_duration - sampledur, file_duration * SAMPLE_LOCATION))
                next_sample = sfile.sound[int(samplestart * 1000):int((samplestart + sampledur) * 1000)]

                # Clear up spacebar capture queue
                while 1:
                    try:
                        spacebarqueue.get_nowait()
                    except queue.Empty:
                        break

                # Gather spacebar press timings and each time, analyze the beats to check whether they are good enough
                # to estimate the BPM accurately
                beat_times = np.array([], dtype='f')
                found_bpm = None
                print(sfile.filepath.encode('utf8').decode(sys.stdout.encoding))
                while found_bpm is None:
                    if samplestart > file_duration or len(next_sample) == 0:
                        break
                    playingsample = play_sample(next_sample)

                    # Get next sample ready for immediate playback
                    samplestart += sampledur
                    millisecond_start = int(samplestart * 1000)
                    millisecond_end = int((samplestart + sampledur) * 1000)
                    next_sample = sfile.sound[millisecond_start:millisecond_end]

                    while playingsample.is_playing():
                        if io.shouldstop:
                            break

                        try:
                            tm = spacebarqueue.get_nowait()
                        except queue.Empty:
                            tm = None

                        if tm == -1: # Special case where user doesn't think this song should have a BPM
                            found_bpm = "Unknown"
                            print("Skipping song permanently and writing 'Unknown' to bpm tag")
                            playingsample.stop()
                            playingsample.wait_done()
                            break

                        elif tm is not None:
                            beat_times = np.append(beat_times, tm - start_tm)
                            tmp_bpm = calc_bpm(beat_times[-2:])

                            # Analyze the spacebar presses and get a BPM estimate and a score as to how good it is
                            bpm, score, other = estimate_bpm_complex(beat_times)

                            # Currently score is binary and anything above 0 means it should be used and is accurate
                            if bpm is not None and score >= CUTOFF_SCORE:
                                found_bpm = bpm
                                playingsample.stop()
                                playingsample.wait_done()
                                break

                        time.sleep(.001)
                    if io.shouldstop:
                        break

                if playingsample.is_playing():
                    print("Stopping song forcefully")
                    playingsample.stop()
                    playingsample.wait_done()

                if found_bpm is not None:
                    try:
                        found_bpm = round(found_bpm)
                    except:
                        pass
                    print(f"BPM({found_bpm})\t{round(time.time() - start_tm, 1):g}s\t{chr(9).join(map(str, [calc_bpm(x, y) for x, y in zip(beat_times[:-1], beat_times[1:])]))}")

                    # Make changes to the file system and add a BPM tag to the mp3 file
                    write_bpm_to_file(sfile, found_bpm)

                else:
                    print(f"WARNING: Stopping and going to next song {sfile.filepath}")
            else:
                print(f'WARNING: Song file is really short or really long {sfile.filepath}')

        io.isrunning = False
    except Exception as e:
        print(e)
        raise e

class SoundFile:
    filepath = None
    id3 = None
    sound = None

    class SoundFileException(Exception):
        pass

    def __init__(self, filepath):
        if not Path(filepath).exists():
            raise self.SoundFileException("ERROR: filepath does not exist")
        self.filepath = filepath
        try:
            self.id3 = EasyID3(self.filepath)
        except mutagen.id3._util.ID3NoHeaderError:
            self.id3 = EasyID3()
            self.id3.filename = str(self.filepath)

    def loadSound(self):
        self.sound = AudioSegment.from_mp3(self.filepath)

    def unloadSound(self):
        if self.sound is not None:
            del self.sound
        self.sound = None

def spaceCapture(key):
    global spacebarqueue
    if key == Key.space:
        spacebarqueue.put(time.time())
    elif key == Key.esc:
        spacebarqueue.put(-1)


class threadIO:
    shouldstop = None
    isrunning = None
    def __init__(self):
        self.shouldstop = False
        self.isrunning = False
            
def main(args=None):
    processing_path = None
    try:
        if args is None or len(args) < 2:
            processing_path = Path(songpath)
        else:
            processing_path = Path(' '.join(sys.argv[1:]))
    except:
        print(f'ERROR: Could not process path "{args}" or "{songpath}"')
    if processing_path is not None:
        print(f"Processing files within {processing_path}")
        if not processing_path.exists():
            raise Exception(f"ERROR: path does not exist {processing_path}")

        with Listener(on_press = spaceCapture) as kc:
            song_playerIO = threadIO()
            try:
                for criteria_func in [lambda x: not has_bpm(x), lambda x: not has_BPMAdder_bpm(x)]:
                    song_files = list(processing_path.rglob("*.mp3"))
                    # random.shuffle(song_files)
                    for path in song_files:
                        r, f, suff = path.parent, path.name, path.suffix
                        if suff.lower() == ".mp3" and criteria_func(r / f):
                            try:
                                soundFile = SoundFile(str(r / f))
                            except SoundFile.SoundFileException as e:
                                print(e)
                                print(f"Failed to interpret soundfile {r / f}")
                                soundFile = None
                            if soundFile is not None:
                                soundFile.loadSound()
                                while 1:
                                    try:
                                        filequeue.put_nowait(soundFile)
                                        break
                                    except queue.Full:
                                        while filequeue.qsize() > 2:
                                            time.sleep(.1)
                                if ENABLE_THREADING:
                                    if not song_playerIO.isrunning:
                                        song_playerIO.isrunning = True
                                        song_player_t = threading.Thread(target=song_player, args=(song_playerIO,))
                                        song_player_t.start()
                                else:
                                    song_player(song_playerIO)
            except KeyboardInterrupt:
                if song_playerIO.isrunning:
                    song_playerIO.shouldstop = True
                else:
                    simpleaudio.stop_all()

            # Wait 30 secs for thread to process each song
            last_size = filequeue.qsize()
            last_time = time.time()
            while song_playerIO.isrunning:
                if last_size != filequeue.qsize():
                    # If we are here we are still making progress
                    last_size = filequeue.qsize()
                    last_time = time.time()
                if time.time() - last_time > 30:
                    # No progress made, try to exit thread
                    song_playerIO.shouldstop = True
                    print("WARNING: Forcefully ending music player thread.")
                    last_time = time.time()
                    while time.time() - last_time < 5:
                        if not song_playerIO.isrunning:
                            break
                        time.sleep(.01)
                    if not time.time() - last_time < 15:
                        print("ERROR: song_player failed to exit.")
                        break

if __name__ == "__main__":
    main()
    input("Done")
