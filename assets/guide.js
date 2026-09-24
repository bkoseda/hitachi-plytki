(() => {
  'use strict';
  const D=window.PING_DEMO, C=['#117c83','#cb6739','#75619b'];
  let step=0, pairView='spectrum';
  const $=id=>document.getElementById(id);
  const layout=(x,y)=>({paper_bgcolor:'transparent',plot_bgcolor:'transparent',margin:{l:64,r:20,t:14,b:48},
    font:{family:'-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',size:11,color:'#405657'},
    xaxis:{title:{text:x,font:{size:11}},gridcolor:'#e0e5de',zeroline:false},
    yaxis:{title:{text:y,font:{size:11}},gridcolor:'#e0e5de',zeroline:false},
    legend:{orientation:'h',x:0,y:1.15},showlegend:true,hovermode:'x unified',
    modebar:{remove:['lasso2d','select2d']}});
  const config={responsive:true,displaylogo:false,displayModeBar:false};
  const draw=(id,data,l)=>Plotly.react($(id),data,{...l,width:$(id).clientWidth,height:$(id).clientHeight},config);
  function line(x,y,name,color){return{x,y,name,type:'scatter',mode:'lines',line:{color,width:1.6}};}
  function active(group,btn){document.querySelectorAll(group).forEach(b=>{b.classList.toggle('active',b===btn);b.setAttribute('aria-pressed',String(b===btn));});}
  function pair(){
    let data,l,caption;
    $('spec-picker').hidden=pairView!=='spectrogram';
    if(pairView==='spectrum'){
      data=D.samples.map((s,i)=>line(s.f,s.spectrum,s.name,C[i]));l=layout('Częstotliwość [Hz]','Moc [dB względem szczytu]');l.xaxis.type='log';l.xaxis.tickvals=innerWidth<600?[100,1000,10000]:[100,300,1000,3000,10000,20000];l.xaxis.ticktext=innerWidth<600?['100','1k','10k']:['100','300','1k','3k','10k','20k'];l.yaxis.range=[-70,2];
      caption='Wyższe piki to rezonanse. Ich położenie mówi, jakie częstotliwości dominują. Obie krzywe mają osobne odniesienie 0 dB.';
    }else if(pairView==='decay'){
      data=D.samples.map((s,i)=>line(s.decay_t,s.decay_db,s.name,C[i]));l=layout('Czas od początku ogona [s]','Energia [dB względem maksimum]');l.yaxis.range=[-65,2];
      caption='Szybszy spadek oznacza krótsze wybrzmiewanie w tym oknie. To może być trop wady, ale także efekt podparcia lub tła.';
    }else if(pairView==='wave'){
      data=D.samples.map((s,i)=>line(s.t,s.y,s.name,C[i]));l=layout('Czas w wyciętym oknie [s]','Amplituda / szczyt');
      caption='Przebieg pokazuje atak i ogon. Wyrównanie szczytów pomaga porównać kształt, lecz usuwa informację o głośności.';
    }else{
      const s=D.samples[Number($('spec-sample').value)];
      data=[{type:'heatmap',x:s.spec_t,y:s.spec_f,z:s.spec_z,zmin:-70,zmax:0,colorscale:'Magma',colorbar:{title:{text:'dB rel.'},thickness:12}}];
      l=layout('Czas w wyciętym oknie [s]','Częstotliwość [Hz]');l.showlegend=false;l.yaxis.range=[0,20000];l.hovermode='closest';
      caption='Pionowy ślad to atak. Poziome pasma to rezonanse, które wybrzmiewają. Jaśniej = więcej energii względem maksimum tej próbki.';
    }
    draw('pair-chart',data,l);$('pair-caption').textContent=caption;
    $('pair-chart').setAttribute('aria-label',caption);
  }
  function hits(){
    const wrong=$('bad-hit').checked,times=[...D.long.onsets];if(wrong)times.splice(1,0,.85);
    const l=layout('Czas nagrania [s]','Amplituda');l.showlegend=false;l.yaxis.range=[-.8,.8];
    l.shapes=times.map((t,i)=>({type:'line',x0:t,x1:t,y0:0,y1:1,yref:'paper',line:{color:wrong&&i===1?'#bb3e32':C[1],width:1.5,dash:'dot'}}));
    l.annotations=times.map((t,i)=>({x:t,y:1,yref:'paper',text:wrong&&i===1?'fałszywy':String(i+1),showarrow:false,font:{size:11,color:wrong&&i===1?'#bb3e32':C[1]},yshift:6}));
    draw('hits-chart',[line(D.long.t,D.long.lo,'minimum',C[0]),line(D.long.t,D.long.hi,'maksimum',C[0])],l);
    $('hit-count').textContent=`${times.length} kandydatów${wrong?' · jeden fałszywy':''}`;
    $('hit-times').textContent='['+times.map(t=>t.toFixed(3)).join(', ')+']';
  }
  function pca(){
    const family=$('feature-family').value,key=$('pca-color').value,table=D.pca[family];
    const groups=[...new Set(table.map(r=>r[key]))];
    const data=groups.map((g,i)=>{const rows=table.filter(r=>r[key]===g);return{type:'scatter',mode:'markers',name:g,
      x:rows.map(r=>r.PC1),y:rows.map(r=>r.PC2),text:rows.map(r=>`${r.plate_id} · ${r.label.toUpperCase()} · ${r.session_id}`),
      marker:{size:11,color:C[i%C.length],symbol:rows.map(r=>r.label==='ok'?'circle':'diamond'),line:{color:'#f8f8f2',width:1}},
      hovertemplate:'%{text}<extra></extra>'};});
    const l=layout('PC1 [jednostki umowne]','PC2 [jednostki umowne]');l.hovermode='closest';
    draw('pca-chart',data,l).then(()=>{
      $('pca-chart').removeAllListeners('plotly_click');
      $('pca-chart').on('plotly_click',e=>{$('pca-caption').textContent=e.points[0].text+' · mediana trzech uderzeń · dane syntetyczne';});
    });
  }
  const methods={
    baseline:'<strong>Zacznij od tej metody.</strong><p>Mediana trzech cech (pik, centroid, zanik) na płytkę + mały klasyfikator. Z góry ustalona regularyzacja. Program nie policzy skuteczności z dwóch płytek.</p><p>Mediana może ukryć lokalną wadę. Jeśli operator słyszy ją tylko w jednym miejscu, analizuj to miejsce osobno.</p>',
    normal:'<strong>Gdy brakuje wadliwych przykładów.</strong><p>Porównaj nową płytkę z co najmniej trzema potwierdzonymi dobrymi wzorcami. Duża odległość oznacza „nietypowe” — przyczyną może być też inny sposób nagrania.</p><p>Próg z tak małej próbki jest roboczy. Nie daje gwarantowanej liczby fałszywych alarmów.</p>',
    ast:'<strong>Inny opis tego samego uderzenia.</strong><p>Zamrożony model AST, wytrenowany na AudioSet, daje 768 liczb. Porównujemy je tym samym małym klasyfikatorem i na tych samych podziałach płytek.</p><p>AST pracuje przy 16 kHz: nie widzi rezonansów powyżej 8 kHz. Nie ma gwarancji, że ogólna wiedza o audio obejmuje wady ceramiki.</p>'};
  function noise(){
    const i=Number($('noise-level').value),s=D.noise[i],base=D.noise[0];
    $('noise-label').textContent=['Bez dodatku szumu','SNR 20 dB','SNR 10 dB','SNR 0 dB'][i];
    const l=layout('Czas od początku ogona [s]','Energia [dB względem maksimum]');l.yaxis.range=[-65,2];
    draw('noise-chart',[line(base.t,base.db,'Oryginał',C[0]),line(s.t,s.db,i===0?'Bez zmiany':'Z dodanym tłem',C[1])],l);
    $('noise-audio').src=`assets/noise-${i}.wav`;
    $('noise-caption').textContent=`Środek widma: ${Math.round(s.centroid)} Hz · czas spadku o 10 dB: ${s.decay.toFixed(2)} s. SNR jest ustawiony dla tego okna. Odsłuch ma wyrównany szczyt.`;
  }
  const hashes=['start','para','uderzenia','cechy','model','halas','wtorek'];
  function go(n){
    step=Math.max(0,Math.min(6,n));document.querySelectorAll('[data-page]').forEach(p=>p.hidden=Number(p.dataset.page)!==step);
    document.querySelectorAll('#nav button').forEach(b=>{const on=Number(b.dataset.step)===step;b.classList.toggle('active',on);if(on)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
    $('page-count').textContent=`0${step+1} / 07`;$('prev').disabled=step===0;$('next').disabled=step===6;
    $('progress').innerHTML=Array.from({length:7},(_,i)=>`<span class="${i<=step?'done':''}"></span>`).join('');
    history.replaceState(null,'','#'+hashes[step]);
    if(step===1)pair();if(step===2)hits();if(step===3)pca();if(step===5)noise();
    window.scrollTo({top:0,behavior:'instant'});
  }
  let toastTimer;
  async function copy(text){
    try{await navigator.clipboard.writeText(text);}catch(_){const t=document.createElement('textarea');t.value=text;document.body.append(t);t.select();document.execCommand('copy');t.remove();}
    $('toast').textContent='Skopiowano';$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),1800);
  }
  document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>go(Number(b.dataset.step)));
  document.querySelectorAll('[data-next]').forEach(b=>b.onclick=()=>go(step+1));
  $('prev').onclick=()=>go(step-1);$('next').onclick=()=>go(step+1);
  document.querySelectorAll('[data-copy]').forEach(b=>b.onclick=()=>copy(b.dataset.copy));
  $('copy-hits').onclick=()=>copy($('hit-times').textContent);$('copy-feedback').onclick=()=>copy($('feedback').value);
  document.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>{active('[data-stage]',b);$('stage-text').innerHTML=b.dataset.stage==='pair'?'<h3>Porównuj. Jeszcze nie oceniaj modelu.</h3><p>Możesz sprawdzić jakość, barwę i rezonanse. Nie możesz wyznaczyć skuteczności detekcji z jednej pary.</p>':'<h3>Sprawdź nowe płytki i nową sesję.</h3><p>Użyj prostych cech, małego klasyfikatora i testu z rozdzieleniem fizycznych detali. Powtórzenia tej samej płytki zostają razem.</p>';});
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{pairView=b.dataset.view;active('[data-view]',b);pair();});
  $('spec-sample').onchange=pair;$('bad-hit').onchange=hits;$('feature-family').onchange=pca;$('pca-color').onchange=pca;
  document.querySelectorAll('[data-method]').forEach(b=>b.onclick=()=>{active('[data-method]',b);$('method-copy').innerHTML=methods[b.dataset.method];});
  $('method-copy').innerHTML=methods.baseline;$('noise-level').oninput=noise;
  $('train-tiles').innerHTML=['P01','P02','P03','P04','P05','P06'].map(v=>`<span class="tile">${v}</span>`).join('');
  $('test-tiles').innerHTML=['P07','P08','P09'].map(v=>`<span class="tile test">${v}</span>`).join('');
  $('checklist').onchange=()=>{$('check-status').textContent=`${$('checklist').querySelectorAll('input:checked').length} / 6 gotowe`;};
  document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause();})));
  window.addEventListener('hashchange',()=>go(Math.max(0,hashes.indexOf(location.hash.slice(1)))));
  document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT','AUDIO','BUTTON'].includes(e.target.tagName))return;if(e.key==='ArrowRight')go(step+1);if(e.key==='ArrowLeft')go(step-1);});
  window.addEventListener('resize',()=>{if(step===1)pair();if(step===2)hits();if(step===3)pca();if(step===5)noise();});
  go(Math.max(0,hashes.indexOf(location.hash.slice(1))));
})();
