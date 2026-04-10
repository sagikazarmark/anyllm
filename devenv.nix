{ pkgs, ... }:

{
  dotenv.enable = true;

  packages = with pkgs; [
    lld

    cargo-release
    cargo-watch
    cargo-expand
  ];

  languages = {
    rust = {
      enable = true;
    };
  };
}
