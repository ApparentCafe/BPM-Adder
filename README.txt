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