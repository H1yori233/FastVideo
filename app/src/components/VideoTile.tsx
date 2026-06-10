import type { Workspace } from "../lib/dataStore";
import type { OutputVideo } from "../lib/types";
import { useInViewVideo } from "./useInView";

export function VideoTile({
  workspace,
  video,
  mode = "static",
}: {
  workspace: Workspace;
  video: OutputVideo | undefined;
  mode?: "auto" | "static";
}) {
  const ref = useInViewVideo();

  if (!video || !video.exists) {
    return (
      <span className="grid h-full w-full place-items-center bg-[repeating-linear-gradient(45deg,#15151b,#15151b_8px,#101015_8px,#101015_16px)]">
        <span className="text-[10px] uppercase tracking-wider text-[#6a6a78]">not generated</span>
      </span>
    );
  }

  const poster = workspace.posterSrc(video);

  if (mode === "auto") {
    return (
      <video
        ref={ref}
        className="block h-full w-full object-cover"
        src={workspace.mediaSrc(video)}
        poster={poster}
        muted
        loop
        playsInline
        preload="metadata"
      />
    );
  }

  if (poster) {
    return (
      <img
        className="block h-full w-full object-cover"
        src={poster}
        loading="lazy"
        decoding="async"
        alt=""
      />
    );
  }

  return (
    <span className="grid h-full w-full place-items-center text-base text-[#4a4a56]">▶</span>
  );
}
