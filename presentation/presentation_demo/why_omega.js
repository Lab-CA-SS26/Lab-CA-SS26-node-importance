// Why the k=0 demo always runs to omega: early-stop counts with the default burn-in, a 10x
// burn-in and a perfect (exact-betweenness) calibration, plus per-vertex samples needed.
// Uses the engine inside betweenness_demo.html (verified against src/kadabra.jl). Run: node why_omega.js
const fs=require('fs');
const src=fs.readFileSync(require('path').join(__dirname,'betweenness_demo.html'),'utf8').match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
const m={exports:{}}; new Function('module',src)(m); const E=m.exports;
const SEEDS=100, delta=0.1;
// run to stop; opts: startFactor, oracle (calibrate deltas from exact bc instead of burn-in)
function run(G, ex, eps, seed, {startFactor=100, oracle=false}={}) {
  const K=new E.Kadabra(G,eps,delta,0,seed,startFactor);
  while(K.phase==='burn') K.step();
  if (oracle) { // same bisection, but fed the exact betweenness as if the burn-in had been perfect
    const order=[...Array(G.n).keys()].sort((a,b)=>ex[b]-ex[a]||a-b);
    const cnt=ex.map(b=>b*1e9);
    const r=E.computeDeltaGuess(cnt,order,1e9,G.n,G.n,true,eps,delta,startFactor);
    K.dL=r.dL; K.dU=r.dU;
  }
  while(K.phase!=='done') K.step();
  return K;
}
for (const key of ['slide','demo']) {
  const G=E.buildGraph(key), ex=E.exactBetweenness(G);
  console.log(`\n== ${key}: n=${G.n}, max b=${Math.max(...ex).toFixed(3)}, VD bound=${E.estimateDiameter(G)}`);
  for (const eps of [0.1,0.05,0.03,0.02]) {
    for (const [name,opt] of [['default',{}],['burn-in x10 (start_factor 10)',{startFactor:10}],['oracle calibration',{oracle:true}]]) {
      let early=0, frac=0;
      for(let s=1;s<=SEEDS;s++){const K=run(G,ex,eps,s,opt); if(K.stopReason==='adaptive')early++; frac+=K.nPairs/K.omega;}
      console.log(`eps=${eps} ${name.padEnd(30)} early stops ${early}/${SEEDS}, mean samples/omega ${(frac/SEEDS).toFixed(2)}`);
    }
  }
  // which vertices bind: with the exact b and oracle deltas, samples needed per vertex for f,g<eps, eps=0.05
  const eps=0.05, K=new E.Kadabra(G,eps,delta,0,1);
  const order=[...Array(G.n).keys()].sort((a,b)=>ex[b]-ex[a]||a-b);
  const r=E.computeDeltaGuess(ex.map(b=>b*1e9),order,1e9,G.n,G.n,true,eps,delta,100);
  const rows=order.map(v=>[G.labels[v], ex[v], Math.log(1/r.dU[v]), E.samplesNeeded(ex[v],1,r.dL[v],r.dU[v],K.omega,eps,100000)]);
  console.log(`oracle, eps=${eps}, omega=${K.omega.toFixed(0)}, omega bracket=${(Math.log2(K.diam-1)+1+Math.log(0.5/delta)).toFixed(2)}`);
  console.log(rows.map(([l,b,L,t])=>`${l}: b=${b.toFixed(3)} ln(1/dU)=${L.toFixed(1)} needs ${t} (${(t/K.omega).toFixed(2)} omega)  4bL=${(4*b*L).toFixed(1)}`).join('\n'));
}
