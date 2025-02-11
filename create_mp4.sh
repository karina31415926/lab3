# load ffmpeg if not already loaded
module list > loaded_modules.txt
grep ffmpeg loaded_modules.txt
FFMPEG_LOADED=$?

if [[ ${FFMPEG_LOADED} -eq 0 ]]; then
        echo "ffmpeg already loaded." ;
else
        echo "loading module ffmpeg" ;
        module load ffmpeg
fi
rm -f loaded_modules.txt

# generate mp4 video from bitmaps
ffmpeg -framerate 30 -pattern_type glob -i '*.bmp'  -pix_fmt yuv420p out.mp4