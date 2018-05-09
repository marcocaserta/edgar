for y in $(seq 2014 1 2014)
do
    echo "***** Year = $y *****"
    echo " "
    echo " "
    python clustering.py -t 2 -y $y
done
