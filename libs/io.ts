import * as _ from "lodash"
import * as fs from 'fs';

export namespace IO {
  export function save(out_path: string, data: any) {
    fs.writeFileSync(out_path, data);
    console.log(`file generated: ${out_path}`);
  }
};