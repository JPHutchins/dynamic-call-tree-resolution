{
  description = "dynamic-call-tree-resolution development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    jphfmt = {
      url = "github:JPHutchins/jphfmt/v0.2.2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, jphfmt, ... }:
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
        in
        {
          default = (pkgs.mkShell.override { stdenv = pkgs.gcc13Stdenv; }) {
            packages = [
              pkgs.uv
              pkgs.git-lfs
              pkgs.gcc-arm-embedded-14
              pkgs.qemu
              pkgs.nixfmt
              jphfmt.packages.${system}.default
            ];
            hardeningDisable = [
              "fortify"
              "fortify3"
            ];
            env.UV_PYTHON_PREFERENCE = "only-managed";
          };
        }
      );
    };
}
