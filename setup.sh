
Option=$1

if [ -z "$1" ]; then
    Option=RELEASE
fi

echo ".. Launching a ${Option} build .."
mkdir -p build-${Option}

cmake -S . -B build-${Option} -DCMAKE_BUILD_TYPE=${Option}
cmake --build build-${Option} -j 4
cmake --install build-${Option} --config ${Option}

ARCH=$(uname -i)

TARGET=$( gcc -v 2>&1 >/dev/null | egrep -o "Target: [0-9a-zA-Z_-]+" | cut -d' ' -f2 | uniq )
VERSION=$( nvcc --version | egrep -o "[0-9]+\.[0-9]+\.[0-9]+" | uniq )

ARCH_STR="${TARGET}-${VERSION}"

pushd build-${Option}

export PATH=$PATH:${PWD}/${ARCH_STR}/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/${ARCH_STR}/lib/
export PYTHONPATH=$PYTHONPATH:${PWD}/${ARCH_STR}/lib/

popd

export PYTHONPATH=$PYTHONPATH:./scripts/
