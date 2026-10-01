{
  description = "dynamic-call-tree-resolution development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    jphfmt = {
      url = "github:JPHutchins/jphfmt/v0.2.2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    zephyr-nix = {
      url = "github:nix-community/zephyr-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    west2nix = {
      url = "github:adisbladis/west2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.zephyr-nix.follows = "zephyr-nix";
    };
  };

  outputs =
    {
      nixpkgs,
      jphfmt,
      zephyr-nix,
      west2nix,
      ...
    }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
      ];
      testbeds =
        pkgs:
        let
          lock = pkgs.lib.importTOML ./testbeds/west2nix.toml;
          zephyrProject = pkgs.lib.findFirst (
            project: project.name == "zephyr"
          ) (throw "testbeds/west2nix.toml locks no zephyr project") lock.manifest.projects;
        in
        {
          sdk = (zephyr-nix.lib.mkZephyr { inherit pkgs; }).sdks."1_0".sdk.override {
            targets = [ "arm-zephyr-eabi" ];
          };
          west2nix = west2nix.lib.mkWest2nix { inherit pkgs; };
          zephyr = zephyr-nix.lib.mkZephyr {
            inherit pkgs;
            zephyr-src = pkgs.fetchgit {
              inherit (zephyrProject) url;
              rev = zephyrProject.revision;
              inherit (zephyrProject.nix) hash;
            };
          };
          buildVersion = builtins.substring 0 12 zephyrProject.revision;
        };
      fixtures =
        pkgs:
        let
          toolchain = testbeds pkgs;
        in
        pkgs.callPackage ./testbeds/fixtures.nix {
          inherit (toolchain) sdk buildVersion;
          pythonEnv = toolchain.zephyr.pythonEnv.override {
            packageOverrides = _: _: { tree-sitter-cmake = null; };
          };
          stdenv = pkgs.multiStdenv;
          west2nixHook = toolchain.west2nix.mkWest2nixHook { manifest = ./testbeds/west2nix.toml; };
          builds = pkgs.lib.importJSON ./testbeds/builds.json;
          src = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./testbeds/manifest
              ./testbeds/.west
              ./tests/fixtures/sensor-two-impl-app
              ./tests/fixtures/sensor-threads-app
            ];
          };
        };
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          devShell =
            stdenv: env: packages:
            (pkgs.mkShell.override { inherit stdenv; }) {
              packages = [
                pkgs.uv
                pkgs.git-lfs
                pkgs.gcc-arm-embedded-14
                pkgs.qemu
                pkgs.nixfmt
                jphfmt.packages.${system}.default
              ]
              ++ packages;
              hardeningDisable = [
                "fortify"
                "fortify3"
              ];
              env = {
                UV_PYTHON_PREFERENCE = "only-managed";
              }
              // env;
            };
        in
        {
          default = devShell pkgs.gcc13Stdenv (nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isx86_64 {
            DCTR_FIXTURES = "${fixtures pkgs}";
          }) [ ];
        }
        // nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isx86_64 {
          testbeds =
            let
              toolchain = testbeds pkgs;
            in
            (devShell pkgs.multiStdenv { USE_CCACHE = "0"; } [
              toolchain.sdk
              toolchain.west2nix.west2nix
              pkgs.dtc
            ]).overrideAttrs
              { hardeningDisable = [ "all" ]; };
        }
      );

      packages.x86_64-linux.fixtures = fixtures nixpkgs.legacyPackages.x86_64-linux;
    };
}
