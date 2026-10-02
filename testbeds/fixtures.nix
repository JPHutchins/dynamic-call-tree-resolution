{
  lib,
  stdenv,
  sdk,
  pythonEnv,
  west2nixHook,
  cmake,
  ninja,
  dtc,
  gitMinimal,
  glibc,
  removeReferencesTo,
  src,
  builds,
  buildVersion,
}:
stdenv.mkDerivation {
  name = "dctr-fixtures";
  inherit src;
  sourceRoot = "source/testbeds";

  nativeBuildInputs = [
    sdk
    pythonEnv
    west2nixHook
    cmake
    ninja
    dtc
    gitMinimal
    removeReferencesTo
  ];

  hardeningDisable = [ "all" ];
  dontUseCmakeConfigure = true;
  dontUseWestConfigure = true;
  dontFixup = true;
  allowedReferences = [ ];

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    mkdir -p build
  ''
  + lib.concatMapStrings (build: ''
    ZEPHYR_TOOLCHAIN_VARIANT=${build.TOOLCHAIN} west build -b ${build.BOARD} -d build/${build.NAME} \
      -s ${build.SOURCE} -- ${lib.escapeShellArg "-DEXTRA_CFLAGS=${build.EXTRA_CFLAGS}"} \
      ${lib.escapeShellArg "-DEXTRA_LDFLAGS=${build.EXTRA_LDFLAGS}"} \
      -DBUILD_VERSION=${buildVersion} > build/${build.NAME}.log 2>&1 \
      || { cat build/${build.NAME}.log; exit 1; }
  '') builds
  + ''
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
  ''
  + lib.concatMapStrings (
    build:
    ''
      (cd build/${build.NAME} && find . \( -name '*.su' -o -name '*.ci' \) \
        -exec install -D -m 644 {} $out/${build.NAME}/{} \;)
      install -D build/${build.NAME}/zephyr/zephyr.exe -t $out/${build.NAME}/zephyr \
        || install -D build/${build.NAME}/zephyr/zephyr.elf -t $out/${build.NAME}/zephyr
    ''
    + lib.optionalString (lib.hasInfix "--print-gc-sections" build.EXTRA_LDFLAGS) ''
      install -D -m 644 build/${build.NAME}/zephyr/zephyr_final.map -t $out/${build.NAME}/zephyr
      sed -n '/Linking C executable zephyr\/zephyr.elf$/,$p' build/${build.NAME}.log \
        | sed -n 's|^.*/ld\.bfd: \(removing unused section .*\)$|\1|p' \
        > $out/${build.NAME}/zephyr/gc-sections.txt
    ''
  ) builds
  + ''
    find $out -type f -exec remove-references-to -t $out -t ${sdk} -t ${stdenv.cc.cc} \
      -t ${stdenv.cc.cc.lib} -t ${stdenv.cc.libc} -t ${glibc.dev} {} +
    runHook postInstall
  '';
}
