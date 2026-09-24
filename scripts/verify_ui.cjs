const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '/Users/karolsii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const path=require('path');
(async()=>{
 const root=path.resolve(__dirname,'..');const out=path.join(root,'reports/ui');fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath:'/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8765/');await page.waitForFunction(()=>window.PING_DEMO);
 const results=[];
 for(let step=0;step<7;step++){
  await page.locator(`#nav button[data-step="${step}"]`).click();
  await page.locator(`[data-page="${step}"]`).waitFor({state:'visible'});
  if([1,2,3,5].includes(step))await page.waitForFunction(()=>document.querySelector('section:not([hidden]) .js-plotly-plot'));
  await page.screenshot({path:path.join(out,`desktop-${step}.png`),fullPage:true,animations:'disabled'});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  if(overflow)throw Error('desktop overflow '+step);
  results.push({test:'desktop step '+step,status:'passed'});
 }
 await page.locator('[data-step="1"]').click();
 for(const view of ['decay','wave','spectrogram','spectrum']){
  await page.locator(`[data-view="${view}"]`).click();
  await page.waitForFunction(v=>document.querySelector(`[data-view="${v}"]`).getAttribute('aria-pressed')==='true',view);
  if(view==='spectrogram'){
    await page.selectOption('#spec-sample','1');
    if(await page.evaluate(()=>document.getElementById('pair-chart').data[0].type)!=='heatmap')throw Error('spectrogram not rendered');
  }
 }
 await page.locator('[data-step="2"]').click();await page.check('#bad-hit');
 if(!(await page.textContent('#hit-count')).includes('4'))throw Error('false-hit interaction');
 await page.uncheck('#bad-hit');
 await page.locator('[data-step="3"]').click();await page.selectOption('#feature-family','mfcc');await page.selectOption('#pca-color','session_id');
 await page.waitForFunction(()=>document.getElementById('pca-chart').data.length===3);
 await page.locator('[data-step="4"]').click();await page.locator('[data-method="ast"]').click();
 if(!(await page.textContent('#method-copy')).includes('768'))throw Error('AST card');
 await page.locator('[data-step="5"]').click();await page.locator('#noise-level').fill('3');
 if(!(await page.textContent('#noise-label')).includes('0 dB'))throw Error('noise slider');
 await page.locator('[data-step="6"]').click();await page.locator('#checklist input').first().check();
 if(!(await page.textContent('#check-status')).includes('1 / 6'))throw Error('checklist');
 // Decode local sample assets through the browser audio stack (without speaker output).
 const decoded=await page.evaluate(async()=>{
  const ctx=new AudioContext();const result=[];
  for(const file of ['demo-a.wav','demo-b.wav','noise-3.wav']){
    const b=await ctx.decodeAudioData(await (await fetch('assets/'+file)).arrayBuffer());
    result.push({file,duration:b.duration,channels:b.numberOfChannels});
  }await ctx.close();return result;
 });
 results.push({test:'all interactions and browser audio decode',status:'passed',decoded});
 await page.setViewportSize({width:390,height:844});
 for(let step=0;step<7;step++){
  await page.locator(`#nav button[data-step="${step}"]`).click();
  await page.screenshot({path:path.join(out,`mobile-${step}.png`),fullPage:true,animations:'disabled'});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error('mobile overflow '+step);
 }
 results.push({test:'all steps at 390 px, no horizontal overflow',status:'passed'});
 await page.setViewportSize({width:1440,height:1050});
 await page.goto('http://127.0.0.1:8765/reports/demo/report.html');
 if(!(await page.textContent('body')).includes('DANE SYNTETYCZNE'))throw Error('missing synthetic warning');
 await page.screenshot({path:path.join(out,'report.png'),fullPage:true,animations:'disabled'});
 if(errors.length)throw Error(JSON.stringify(errors));
 fs.writeFileSync(path.join(root,'reports/ui_validation.json'),JSON.stringify({results,javascript_errors:errors},null,2));
 await browser.close();console.log(JSON.stringify(results));
})().catch(e=>{console.error(e);process.exit(1)});
