import { spawn } from "child_process";
import { platform } from "os";
import { resolve } from "path";
import { existsSync } from "fs";

/**
 * ブラウザでファイルを開く（クロスプラットフォーム対応）
 */
export function openInBrowser(filePath: string): boolean {
  const absolutePath = resolve(filePath);

  if (!existsSync(absolutePath)) {
    console.error(`ファイルが見つかりません: ${absolutePath}`);
    return false;
  }

  const currentPlatform = platform();
  let command: string;
  let args: string[];

  switch (currentPlatform) {
    case "darwin": // macOS
      command = "open";
      args = [absolutePath];
      break;
    case "win32": // Windows
      command = "cmd";
      args = ["/c", "start", '""', absolutePath];
      break;
    case "linux":
      command = "xdg-open";
      args = [absolutePath];
      break;
    default:
      console.error(`サポートされていないプラットフォーム: ${currentPlatform}`);
      console.log(`手動で開いてください: file://${absolutePath}`);
      return false;
  }

  try {
    const child = spawn(command, args, {
      detached: true,
      stdio: "ignore",
      shell: currentPlatform === "win32",
    });

    child.unref();

    return true;
  } catch (error) {
    console.error(`ブラウザの起動に失敗しました:`, error);
    console.log(`手動で開いてください: file://${absolutePath}`);
    return false;
  }
}
