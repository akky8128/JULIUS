#!/usr/bin/env node
/**
 * abProbe.mjs — 単発診断: AZネットを「価値ヘッド+αβ探索(ターン文脈ゼロ)」= ブラウザ
 * (nnueWorker.js / analysisWorker.js)と同じモードで対戦させ、実強度を測る。
 * MCTSでの測定強度(28-0 vs gen137)とは別物の、ブラウザ配備モードの検証用。
 *   node abProbe.mjs --netA research/models/genAZ8_r83.json --games 20 --seed 1 \
 *     --oppNet research/models/gen137.json --hdepth 16 --hms 80
 * oppNet 指定時は netA+ab vs oppNet+ab、未指定時は netA+ab vs heuristic。
 */
import { generateTurns } from "../../../core/ai/moveGen.js";
import { normalizeBoard } from "../../../core/gameLogic.js";
import { findBestTurn } from "../../../core/ai/search.js";
import { WEIGHTS } from "../../../core/ai/evaluate.js";
import { loadNetwork } from "../../../core/nnue/network.js";

function mulberry32(seed){return function(){seed|=0;seed=(seed+0x6d2b79f5)|0;let t=Math.imul(seed^(seed>>>15),1|seed);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
const emptyBoard=(s)=>Array.from({length:s},()=>Array.from({length:s},()=>[]));
const nextPlayer=(p)=>p==="white"?"black":"white";
function playGame(w,b,size,maxPlies){let board=normalizeBoard(emptyBoard(size),size),summonCounts={white:0,black:0},cur="white",ply=0;while(ply<maxPlies){const st={board:normalizeBoard(board,size),summonCounts:{...summonCounts},currentPlayer:cur,boardSize:size};if(generateTurns(st).length===0)return nextPlayer(cur);const chosen=(cur==="white"?w:b).chooseTurn(st);board=normalizeBoard(chosen.board,size);summonCounts=chosen.summonCounts;cur=nextPlayer(cur);ply++;}return null;}
const netAbAgent=(net,depth,ms,rng)=>({chooseTurn(state){const res=findBestTurn(state,{maxDepth:depth,timeBudgetMs:ms,evalFn:(s,p)=>net.evaluateState(s,p),rng});return{board:res.turn.board,summonCounts:res.turn.summonCounts};}});
const heuristicAgent=(depth,ms,rng)=>({chooseTurn(state){const res=findBestTurn(state,{maxDepth:depth,timeBudgetMs:ms,weights:WEIGHTS,rng});return{board:res.turn.board,summonCounts:res.turn.summonCounts};}});

async function main(){
  const argv=process.argv.slice(2);const get=(k,d)=>{const i=argv.indexOf(k);return i>=0?argv[i+1]:d;};
  const netAPath=get("--netA","research/models/genAZ8_r83.json");
  const oppNetPath=get("--oppNet",null);
  const games=parseInt(get("--games","20"),10);
  const seed=parseInt(get("--seed","1"),10);
  const hdepth=parseInt(get("--hdepth","16"),10);
  const hms=parseInt(get("--hms","80"),10);
  const sdepth=parseInt(get("--sdepth","32"),10);
  const sms=parseInt(get("--sms","120"),10);
  const size=4,maxPlies=200;const rng=mulberry32(seed);
  const netA=await loadNetwork(netAPath);
  const oppNet=oppNetPath?await loadNetwork(oppNetPath):null;
  const focal=netAbAgent(netA,sdepth,sms,rng);
  const opp=oppNet?netAbAgent(oppNet,sdepth,sms,rng):heuristicAgent(hdepth,hms,rng);
  let wins=0,losses=0,draws=0;
  for(let g=0;g<games;g++){const fWhite=g%2===0;const winner=fWhite?playGame(focal,opp,size,maxPlies):playGame(opp,focal,size,maxPlies);const fc=fWhite?"white":"black";if(winner===null)draws++;else if(winner===fc)wins++;else losses++;}
  const oppLabel=oppNet?`${oppNetPath}+ab(d${sdepth}:${sms})`:`heuristic-d${hdepth}:${hms}`;
  console.log(`[abProbe] ${netAPath}+ab(d${sdepth}:${sms}) vs ${oppLabel}: ${wins}W/${losses}L/${draws}D  (games=${games})`);
}
main().catch((e)=>{console.error(e);process.exitCode=1;});
