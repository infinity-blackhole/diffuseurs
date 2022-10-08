{
  description = "Collection of diffusion model for Axon";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/release-22.05";
  outputs = { self, nixpkgs }: {
    devShell = nixpkgs.lib.genAttrs nixpkgs.lib.platforms.unix (system:
      with import nixpkgs { inherit system; };
      mkShell {
        buildInputs = [
          elixir
          nixpkgs-fmt
        ];
      }
    );
  };
}
