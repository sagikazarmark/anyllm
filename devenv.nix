{ pkgs, ... }:

{
  dotenv.enable = true;

  packages = with pkgs; [
    lld

    cargo-audit
    cargo-expand
    cargo-release
    cargo-watch
  ];

  languages = {
    rust = {
      enable = true;
    };
  };
}
