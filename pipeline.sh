#!/bin/bash

#SBATCH --account=rrg-kmsmith
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=2
#SBATCH --mem=257000M
#SBATCH --time=24:00:00
#SBATCH --job-name=pathfinder
#SBATCH --output=/home/rebeccac/scratch/H-GASP/pathfinder/pipeline.out

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

############################################ Step 0 - set up  #################################################
echo "-------------- Step 0 - Set up --------------"

# loading modules
echo "Loading CHORD pipeline modules..."
module --force purge
module use /project/def-mdobbs/ssiegel/chord/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2022.11

##### COMBINE THESE THREE INTO ONE FILE, NOT NEEDED TO HAVE IN 3 STEPS
# get frequencies and add to outputs.yaml file
echo "Setting up frequency channels..."
python get_freqs.py

# calculate elevation based on observing_dec and CHORD's latitude
echo "Calculating elevation of the observation..."
python get_elevation.py

# set up configuration files for beam transfer matrices and observation computations
echo "Setting up configuration files..."
python change_configs.py

eval $(parse_yaml inputs.yaml)
output_folder=$(echo $process_output_folder)
cd $output_folder

eval $(parse_yaml $output_folder/outputs.yaml)
fmin=$(echo $frequencies_fmin)
fmax=$(echo $frequencies_fmax)
nside=$(echo $telescope_nside)
nfreq_maps=$(echo $frequencies_nfreqmaps)
num=4
fmap_max=$(echo $(expr $fmax + $num))
fmap_min=$(echo $(expr $fmin - $num))

######################################## Step 1 - tool computation  ############################################
echo "-------------- Step 1 - Tool computation --------------"

## should i split it into 2 separete jobs? in terms of more efficiently using the nodes?

# computing the beam transfer matrices
# computing the response matrix R and the normalization vector norm
# set up so that they will both be computed simultaneously
echo "Computing the response matrix and normalization vector..."
echo "Computing the beam transfer matrices..."
source /dev/null
cd /home/rebeccac/scratch/pipeline
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python /project/6002277/ssiegel/chord/chord_env/modules/chord/chord_pipeline/2022.11/lib/python3.10/site-packages/drift/scripts/makeproducts.py run $output_folder/beam.yaml &> $output_folder/beam.log &
python get_response_mtx.py &
wait


######################################## Step 2 - map creation  ############################################
echo "---------------- Step 2 - Map creation ----------------"

# cora makesky map components
echo "Generating sky maps with cora..."
cora-makesky foreground --nside=$nside --freq $fmap_max $fmap_min $nfreq_maps --pol=full --filename=$output_folder/foregrounds.h5
cora-makesky gaussianfg --nside=$nside --freq $fmap_max $fmap_min $nfreq_maps --pol=full --filename=$output_folder/synch_map.h5

# getting upchannelized sky maps
echo "Up-channelizing cora simulated maps..."
python get_upchannelized_map.py 1

# getting upchannelized galaxy profiles
echo "Up-channelizing galaxy catalog..."
python get_upchannelized_map.py 0

######################################## Step 3 - observation  ############################################
echo "----------------- Step 3 - Observation -----------------"

echo "Performing observation with caput..."
srun python /project/6002277/ssiegel/chord/chord_env/modules/chord/chord_pipeline/2022.11/lib/python3.10/site-packages/caput/scripts/runner.py run $output_folder/simulate.yaml &> $output_folder/simulate.log

######################################## Step 4 - Noise and Dirty map  ############################################
echo "--------------- Step 4 - Noise and Dirty map ---------------"

echo "Getting noisy visibilities and noisy dirty map..."
python get_noise.py 1
python get_noise.py 0

echo " "
echo "COMPLETE"
