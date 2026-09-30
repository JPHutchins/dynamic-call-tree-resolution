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
  };

  outputs =
    {
      nixpkgs,
      jphfmt,
      zephyr-nix,
      ...
    }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          devShell =
            stdenv: packages:
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
              env.UV_PYTHON_PREFERENCE = "only-managed";
            };
        in
        {
          default = devShell pkgs.gcc13Stdenv [ ];
        }
        // nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isx86_64 {
          testbeds = devShell pkgs.multiStdenv [
            ((zephyr-nix.lib.mkZephyr { inherit pkgs; }).sdks."1_0".sdk.override {
              targets = [ "arm-zephyr-eabi" ];
            })
            pkgs.dtc
          ];
        }
      );
    };
}
