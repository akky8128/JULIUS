// usage: node agg.mjs "<agentSubstr>" file1 file2 ...
import { readFileSync } from "fs";
const [sub, ...files] = process.argv.slice(2);
let wins = 0, dec = 0;
for (const f of files) {
  let txt; try { txt = readFileSync(f, "utf8"); } catch { continue; }
  for (const line of txt.split("\n")) {
    if (line.includes(sub) && line.includes("勝") && line.includes("決着")) {
      const m = line.match(/([0-9]+)勝 \/ ([0-9]+)決着/);
      if (m) { wins += +m[1]; dec += +m[2]; break; } // first summary line per file
    }
  }
}
if (dec === 0) { console.log(`${sub}: NO DATA`); process.exit(0); }
const p = wins / dec, z = 1.96, n = dec;
const c = (p + z*z/(2*n)) / (1 + z*z/n);
const h = (z * Math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n);
const pc = x => (100*x).toFixed(1);
console.log(`${sub}: ${wins}勝 / ${dec}決着  勝率 ${pc(p)}%  95%CI [${pc(c-h)}, ${pc(c+h)}]`);
