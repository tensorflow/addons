# Downloads bazelisk to ${output_dir} as `bazel`.
date

output_dir=${1:-"/usr/local/bin"}

case "$(uname -s)" in
    Darwin) name=bazelisk-darwin-amd64 ;;
    Linux)  name=bazelisk-linux-amd64  ;;
    *) name=bazelisk-windows-amd64 ;;
esac

mkdir -p "${output_dir}"
curl -LO "https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/${name}"

mv "${name}" "${output_dir}/bazel"
chmod u+x "${output_dir}/bazel"

if [[ ! ":$PATH:" =~ :${output_dir}/?: ]]; then
    PATH="${output_dir}:$PATH"
fi

which bazel
date
