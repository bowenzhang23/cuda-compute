
pushd build

cmake ..
make install -j 4

ARCH=$(uname -i)

TARGET=$( gcc -v 2>&1 >/dev/null | egrep -o "Target: [0-9a-zA-Z_-]+" | cut -d' ' -f2 | uniq )
VERSION=$( nvcc --version | egrep -o "[0-9]+\.[0-9]+\.[0-9]+" | uniq )

ARCH_STR="${TARGET}-${VERSION}"

export PATH=$PATH:${PWD}/${ARCH_STR}/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/${ARCH_STR}/lib/
export PYTHONPATH=$PYTHONPATH:${PWD}/${ARCH_STR}/lib/

popd
