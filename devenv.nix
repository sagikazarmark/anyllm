{ pkgs, ... }:

{
  dotenv.enable = true;

  packages = with pkgs; [
    lld

    cargo-audit
    cargo-deny
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
