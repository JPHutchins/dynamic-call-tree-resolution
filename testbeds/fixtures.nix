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
  nukeReferences,
  gcc-arm-embedded-14,
  descriptorPlugin,
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
    nukeReferences
  ];

  exportReferencesGraph = lib.concatLists (
    lib.imap0
      (index: input: [
        "input-closure-${toString index}"
        input
      ])
      [
        sdk
        pythonEnv
        west2nixHook
        cmake
        ninja
        dtc
        gitMinimal
        nukeReferences
        gcc-arm-embedded-14
        descriptorPlugin
        stdenv.cc
      ]
  );

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
  + lib.concatMapStrings (
    build:
    ''
      ZEPHYR_TOOLCHAIN_VARIANT=${build.TOOLCHAIN} west build -b ${build.BOARD} -d build/${build.NAME} \
        -s ${build.SOURCE} -- ${lib.escapeShellArg "-DEXTRA_CFLAGS=${build.EXTRA_CFLAGS}"} \
        ${lib.escapeShellArg "-DEXTRA_LDFLAGS=${build.EXTRA_LDFLAGS}"} \
        -DBUILD_VERSION=${buildVersion} > build/${build.NAME}.log 2>&1 \
        || { cat build/${build.NAME}.log; exit 1; }
    ''
    + lib.optionalString (build.TOOLCHAIN == "zephyr") ''
      PATH=${gcc-arm-embedded-14}/bin:$PATH python3 plugin/replay.py build/${build.NAME} \
        ${descriptorPlugin}/lib/descriptors.so "$(realpath ..)"
    ''
  ) builds
  + ''
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out
  ''
  + lib.concatMapStrings (
    build:
    ''
      cp -r build/${build.NAME} $out/${build.NAME}
    ''
    + lib.optionalString (lib.hasInfix "--print-gc-sections" build.EXTRA_LDFLAGS) ''
      sed -n '/Linking C executable zephyr\/zephyr.elf$/,$p' build/${build.NAME}.log \
        | sed -n 's|^.*/ld\.bfd: \(removing unused section .*\)$|\1|p' \
        > $out/${build.NAME}/zephyr/gc-sections.txt
    ''
  ) builds
  + ''
    find $out -type f -exec nuke-refs {} +
    { grep -ho '/nix/store/[0-9a-z]\{32\}' $NIX_BUILD_TOP/input-closure-* | cut -c12-; \
      basename $out | cut -c1-32; } | sort -u > $TMPDIR/hashes
    sed 's|.*|s/&/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/g|' $TMPDIR/hashes > $TMPDIR/scrub.sed
    grep -rlF --null -f $TMPDIR/hashes $out | xargs -0 -r sed -i -f $TMPDIR/scrub.sed
    runHook postInstall
  '';
}
