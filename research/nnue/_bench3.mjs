import { generateTurns } from "../../core/ai/moveGen.js";
import { evaluate, WEIGHTS } from "../../core/ai/evaluate.js";
import { findBestTurn } from "../../core/ai/search.js";
import { normalizeBoard, maxSummonsFor } from "../../core/gameLogic.js";
import { loadNetwork } from "../../core/nnue/network.js";
function mul(a){return function(){a|=0;a=a+0x6d2b79f5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const size=4;
const eb=()=>Array.from({length:size},()=>Array.from({length:size},()=>[]));
let rng=mul(999), board=eb(), sc={white:0,black:0}, cur="white";
for(let i=0;i<24;i++){const st={board:normalizeBoard(board,size),summonCounts:{...sc},currentPlayer:cur,boardSize:size};const t=generateTurns(st);if(!t.length)break;const c=t[Math.floor(rng()*t.length)];board=normalizeBoard(c.board,size);sc=c.summonCounts;cur=cur==="white"?"black":"white";}
const state={board:normalizeBoard(board,size),summonCounts:{...sc},currentPlayer:cur,boardSize:size};
console.log(`局面: 分岐=${generateTurns(state).length} 手番=${cur} 盤上駒=${board.flat().reduce((a,s)=>a+s.length,0)}`);
console.log("\n=== ① 予算遵守 & ②③ 同一時間での深さ/ノード (heuristic) ===");
for(const budget of [200,1000]){
  const off=findBestTurn(state,{maxDepth:20,timeBudgetMs:budget,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:0});
  const on =findBestTurn(state,{maxDepth:20,timeBudgetMs:budget,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:48});
  console.log(`予算${budget}ms  OFF: depth=${off.depthReached} nodes=${off.nodes} 実${off.elapsedMs}ms | ON(w48): depth=${on.depthReached} nodes=${on.nodes} 実${on.elapsedMs}ms`);
}
console.log("\n=== 同一深さの所要時間 (heuristic, 時間無制限) ===");
for(const d of [3,4,5]){
  const off=findBestTurn(state,{maxDepth:d,timeBudgetMs:1e9,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:0});
  const on =findBestTurn(state,{maxDepth:d,timeBudgetMs:1e9,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:48});
  console.log(`depth${d}  OFF: ${off.elapsedMs}ms/${off.nodes}nodes | ON: ${on.elapsedMs}ms/${on.nodes}nodes => ${(off.elapsedMs/Math.max(1,on.elapsedMs)).toFixed(1)}x`);
}
const net=await loadNetwork("research/models/gen001.json");
const efn=(s,p)=>net.evaluateState(s,p);
console.log("\n=== NNUE: 同一時間での深さ/ノード ===");
for(const budget of [200,1000]){
  const off=findBestTurn(state,{maxDepth:20,timeBudgetMs:budget,rng:mul(1),evalFn:efn,forwardPruneWidth:0});
  const on =findBestTurn(state,{maxDepth:20,timeBudgetMs:budget,rng:mul(1),evalFn:efn,forwardPruneWidth:48});
  console.log(`予算${budget}ms  OFF: depth=${off.depthReached} nodes=${off.nodes} 実${off.elapsedMs}ms | ON(w48): depth=${on.depthReached} nodes=${on.nodes} 実${on.elapsedMs}ms`);
}
