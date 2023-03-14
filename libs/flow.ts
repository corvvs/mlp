import * as _ from "lodash"
import { exit } from "process"

export namespace Flow {
  export function exit_with_error(message?: string): never {
    if (message) {
      console.error(message);
    }
    exit(1);
  }
}
