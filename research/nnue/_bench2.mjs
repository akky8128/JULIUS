import { generateTurns } from "../../core/ai/moveGen.js";
import { evaluate, WEIGHTS } from "../../core/ai/evaluate.js";
import { findBestTurn } from "../../core/ai/search.js";
import { normalizeBoard, maxSummonsFor, checkWinCondition } from "../../core/gameLogic.js";
function mul(a){return function(){a|=0;a=a+0x6d2b79f5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const size=4, maxS=maxSummonsFor(size);
const eb=()=>Array.from({length:size},()=>Array.from({length:size},()=>[]));
// 代表中盤局面
let rng=mul(999), board=eb(), sc={white:0,black:0}, cur="white";
for(let i=0;i<24;i++){const st={board:normalizeBoard(board,size),summonCounts:{...sc},currentPlayer:cur,boardSize:size};const t=generateTurns(st);if(!t.length)break;const c=t[Math.floor(rng()*t.length)];board=normalizeBoard(c.board,size);sc=c.summonCounts;cur=cur==="white"?"black":"white";}
const state={board:normalizeBoard(board,size),summonCounts:{...sc},currentPlayer:cur,boardSize:size};
console.log(`局面: 分岐=${generateTurns(state).length} 手番=${cur} 盤上駒=${board.flat().reduce((a,s)=>a+s.length,0)}`);

console.log("\n=== ① 予算遵守 & ②③ 同一時間予算での深さ/ノード (heuristic) ===");
for(const budget of [200,1000]){
  const off=findBestTurn(state,{maxDepth:20,timeBudgetMs:budget,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:0});
  const on =findBestTurn(state,{maxDepth:20,timeBudgetMs:budget,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:48});
  console.log(`予算${budget}ms  OFF: depth=${off.depthReached} nodes=${off.nodes} 実${off.elapsedMs}ms | ON(w48): depth=${on.depthReached} nodes=${on.nodes} 実${on.elapsedMs}ms`);
}

console.log("\n=== 同一深さでの所要時間 (heuristic, 時間無制限) ===");
for(const d of [3,4,5]){
  const off=findBestTurn(state,{maxDepth:d,timeBudgetMs:1e9,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:0});
  const on =findBestTurn(state,{maxDepth:d,timeBudgetMs:1e9,weights:WEIGHTS,rng:mul(1),forwardPruneWidth:48});
  console.log(`depth${d}  OFF: ${off.elapsedMs}ms/${off.nodes}nodes | ON: ${on.elapsedMs}ms/${on.nodes}nodes  => ${(off.elapsedMs/Math.max(1,on.elapsedMs)).toFixed(1)}x速`);
}

// 新旧対戦(同一時間予算, heuristic)
function play(size,maxPlies,wOpt,bOpt,seed){
  let board=eb(),sc={white:0,black:0},cur="white",ply=0;const r=mul(seed);
  const wr=mul((r()*4294967295)>>>0), br=mul((r()*4294967295)>>>0);
  while(ply<maxPlies){
    const st={board:normalizeBoard(board,size),summonCounts:{...sc},currentPlayer:cur,boardSize:size};
    const cands=generateTurns(st); if(!cands.length) return cur==="white"?"black":"white";
    const opt=cur==="white"?wOpt:bOpt; const rr=cur==="white"?wr:br;
    const res=findBestTurn(st,{...opt,weights:WEIGHTS,rng:rr});
    const turn=res.turn||cands[0];
    board=normalizeBoard(turn.board,size);sc=turn.summonCounts;cur=cur==="white"?"black":"white";ply++;
  }
  return null;
}
console.log("\n=== ②③ 効果: 新(w48) vs 旧(w0) 同一時間予算 250ms, heuristic ===");
const A={maxDepth:20,timeBudgetMs:250,forwardPruneWidth:48}; // 新
const B={maxDepth:20,timeBudgetMs:250,forwardPruneWidth:0};  // 旧
let aWin=0,bWin=0,draw=0,G=40;
for(let g=0;g<G;g++){
  const swap=g>=G/2; const w=swap?B:A, b=swap?A:B;
  const winner=play(size,100,w,b,1000+g);
  const aColor=swap?"black":"white";
  if(winner===null)draw++; else if(winner===aColor)aWin++; else bWin++;
}
const dec=aWin+bWin, p=aWin/dec, ci=1.96*Math.sqrt(p*(1-p)/dec);
console.log(`新(w48) ${aWin}勝 / 旧(w0) ${bWin}勝 / draw ${draw}  新勝率=${(100*p).toFixed(1)}% ±${(100*ci).toFixed(1)}% (n=${dec})`);
