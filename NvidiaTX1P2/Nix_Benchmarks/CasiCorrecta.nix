# Derivacion casi correcta:
# Error: no encuentra mpi.h
{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation rec {
  name    = "examplempi";

  src = /home/nvidia/nixpkgs/pkgs/misc/examplempi;

  HOME="/home/nvidia";
  NIX_PATH = "/usr/local/cuda/bin:/usr/local/openmpi/bin:/usr/local/openblas/bin";

  
  phases = [ "buildPhase" "installPhase" ];

#  nativeBuildInputs = [ openmpi ];

  buildPhase = ''
    
#    patchelf --set-interpreter "/usr/local/openmpi/bin/mpicc" "$src/mpi_hello_world"

    export PATH="$PATH:/usr/local/openmpi/bin"
    export PATH="$PATH:/usr/local/openblas/bin"

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/openmpi/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/openblas/lib"

    cd $src
    make

  '';

  installPhase = ''

    export PATH="$PATH:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/game"

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib"
1    
 '';
}
