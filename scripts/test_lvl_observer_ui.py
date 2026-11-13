#!/usr/bin/env python3
"""HTTP contract checks for the local LVL observer; does not start GAMA.

Run the observer separately with ``python3 scripts/lvl_observer_server.py`` and
then run this file.  The suite uses only the Python standard library.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("LVL_OBSERVER_URL", "http://127.0.0.1:8765").rstrip("/")
HTML_PATH = Path(__file__).resolve().parents[1] / "python-simulation" / "ui" / "observer.html"


def request(path, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(BASE_URL + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=4) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type:
                return response.status, json.loads(raw)
            return response.status, raw
    except HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = raw
        return exc.code, body


class ObserverHttpContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            code, state = request("/api/state")
        except (URLError, TimeoutError, OSError) as exc:
            raise unittest.SkipTest(
                f"observer unavailable at {BASE_URL}; start it separately ({exc})"
            )
        if code != 200 or not isinstance(state, dict):
            raise unittest.SkipTest(f"/api/state unavailable: HTTP {code}")

    def test_roads_is_nonempty_geojson(self):
        code, roads = request("/api/roads")
        self.assertEqual(code, 200)
        self.assertIsInstance(roads, dict)
        self.assertEqual(roads.get("type"), "FeatureCollection")
        self.assertTrue(roads.get("features"), "map should have road features")

    def test_root_serves_the_observer_page(self):
        code, page = request("/")
        self.assertEqual(code, 200)
        self.assertIsInstance(page, bytes)
        self.assertIn(b"<!doctype html>", page[:64].lower())
        self.assertIn(b'id="step1"', page)
        self.assertIn(b'id="step50"', page)

    def test_state_has_renderable_snapshot_and_vehicle_collection(self):
        code, state = request("/api/state")
        self.assertEqual(code, 200)
        for key in ("cycle", "time_sec", "phase", "created", "passed", "active", "queue"):
            self.assertIn(key, state, f"missing state field used by observer: {key}")
        self.assertIsInstance(state["vehicles"], list)
        self.assertGreaterEqual(state["cycle"], 0)
        self.assertIn(state["phase"], ("LVL", "HDT"))
        self.assertGreaterEqual(state["created"], state["active"])

    def test_one_and_fifty_steps_advance_exact_cycle_counts(self):
        code, initial = request("/api/reset", "POST", {"seed": 8173})
        self.assertEqual(code, 200)
        start_cycle = initial["cycle"]
        code, one = request("/api/step", "POST", {"cycles": 1})
        self.assertEqual(code, 200)
        self.assertEqual(one["cycle"], start_cycle + 1)
        code, fifty = request("/api/step", "POST", {"cycles": 50})
        self.assertEqual(code, 200)
        self.assertEqual(fifty["cycle"], start_cycle + 51)
        self.assertEqual(fifty["time_sec"], round((start_cycle + 51) * 0.2, 3))
        self.assertIsInstance(fifty["vehicles"], list)

    def test_reset_seed_restores_initial_state_deterministically(self):
        first_code, first = request("/api/reset", "POST", {"seed": 1942})
        self.assertEqual(first_code, 200)
        request("/api/step", "POST", {"cycles": 5})
        second_code, second = request("/api/reset", "POST", {"seed": 1942})
        self.assertEqual(second_code, 200)
        self.assertEqual(first, second, "same seed should restore same initial snapshot")

    def test_mode_reset_train_step_wait_and_action_contract(self):
        code, fixed = request("/api/reset", "POST", {"seed": 8173, "mode": "fixed"})
        self.assertEqual(code, 200)
        self.assertEqual(fixed.get("mode"), "fixed")
        code, train = request("/api/reset", "POST", {"seed": 8173, "mode": "train"})
        self.assertEqual(code, 200)
        self.assertEqual(train.get("mode"), "train")

        code, waiting = request("/api/step", "POST", {"cycles": 50})
        self.assertEqual(code, 200)
        self.assertEqual(waiting.get("step_status"), "waiting_for_action")
        self.assertEqual(waiting.get("cycles_advanced"), 50)
        self.assertTrue(waiting.get("dqn_waiting_action"))
        for action in (0, 2):
            with self.subTest(action=action):
                # Return to the same deterministic decision boundary for each action.
                code, _ = request("/api/reset", "POST", {"seed": 8173, "mode": "train"})
                self.assertEqual(code, 200)
                code, state = request("/api/step", "POST", {"cycles": 50})
                self.assertEqual(code, 200)
                self.assertEqual(state.get("step_status"), "waiting_for_action")
                code, chosen = request("/api/action", "POST", {"action": action})
                self.assertEqual(code, 200)
                self.assertFalse(chosen.get("dqn_waiting_action"))
                self.assertEqual(chosen.get("mode"), "train")

        code, _ = request("/api/reset", "POST", {"seed": 8173, "mode": "fixed"})
        self.assertEqual(code, 200)
        code, body = request("/api/action", "POST", {"action": 0})
        self.assertEqual(code, 409)
        self.assertIn("error", body)

    def test_invalid_mode_and_actions_are_rejected(self):
        code, body = request("/api/reset", "POST", {"seed": 1, "mode": "other"})
        self.assertEqual(code, 400)
        self.assertIn("error", body)
        for action in (1, -1, 2.0, "0", None):
            with self.subTest(action=action):
                code, body = request("/api/action", "POST", {"action": action})
                self.assertEqual(code, 400)
                self.assertIn("error", body)

    def test_invalid_step_and_reset_inputs_are_client_errors(self):
        for value in (0, -1, 1.5, "5", 101, None):
            with self.subTest(cycles=value):
                code, body = request("/api/step", "POST", {"cycles": value})
                self.assertEqual(code, 400)
                self.assertIsInstance(body, dict)
                self.assertIn("error", body)
        for value in ("abc", 1.25, None):
            with self.subTest(seed=value):
                code, body = request("/api/reset", "POST", {"seed": value})
                self.assertEqual(code, 400)
                self.assertIsInstance(body, dict)
                self.assertIn("error", body)

    def test_unknown_route_is_not_success(self):
        code, body = request("/api/no-such-route")
        self.assertEqual(code, 404)
        self.assertIsInstance(body, dict)
        self.assertIn("error", body)


class ObserverUiStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HTML_PATH.read_text(encoding="utf-8")

    def test_all_requested_controls_and_speed_values_exist(self):
        for control in ("run", "pause", "step1", "step50", "reset", "seed", "rate", "mode", "actionLvl", "actionHdt", "decisionStatus"):
            with self.subTest(control=control):
                self.assertIn(f'id="{control}"', self.html)
        for rate in ("1", "5", "20"):
            with self.subTest(rate=rate):
                self.assertIn(f'<option value="{rate}"', self.html)

    def test_manual_step_buttons_send_fixed_counts_and_auto_uses_selected_count(self):
        self.assertIn("addEventListener('click',()=>step(1)", self.html)
        self.assertIn("addEventListener('click',()=>step(50)", self.html)
        self.assertIn("await step(Number(el('rate').value))", self.html)
        self.assertIn("JSON.stringify({cycles})", self.html)

    def test_run_pause_seed_reset_and_error_feedback_are_wired(self):
        for expected in (
            "addEventListener('click',run)",
            "addEventListener('click',pause)",
            "const seed=Number(el('seed').value)",
            "'/api/reset'",
            "Number.isSafeInteger(seed)",
            "showError(e.message)",
            "'Mất kết nối máy chủ'",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.html)

    def test_mode_actions_and_waiting_status_are_present(self):
        for expected in ('value="fixed"', 'value="train"', "'/api/action'", "waiting_for_action", "cycles_advanced"):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.html)

    def test_signal_counters_and_vehicles_are_refreshed_from_state(self):
        for element in ("phaseLight", "phaseName", "cycle", "time", "created", "passed", "active", "queue", "vehicleCount"):
            with self.subTest(element=element):
                self.assertIn(f"el('{element}')", self.html)
        self.assertIn("vehicles=data.vehicles||data.state?.vehicles||[]", self.html)
        self.assertIn("drawSignals()", self.html)
        self.assertIn("drawVehicles()", self.html)

    def test_signal_colors_follow_phase_and_approach_arm(self):
        self.assertIn("phase==='LVL'&&lvlArms.includes(road.arm)", self.html)
        self.assertIn("phase==='HDT'&&hdtArms.includes(road.arm)", self.html)
        self.assertIn("lvlArms=['NE','SW']", self.html)
        self.assertIn("hdtArms=['SE','NW']", self.html)
        self.assertIn("return colors['ALL-RED']", self.html)
        self.assertIn("['NONE','ALL-RED','ALL_RED','RED'].includes(phase)", self.html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for mocked DOM interaction tests")
    def test_controls_with_mocked_dom_canvas_and_api(self):
        harness = r'''const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(0,'utf8');
const source=(html.match(/<script>([\s\S]*?)<\/script>/)||[])[1];
assert(source,'inline observer script missing');
const ids=['map','connectionDot','connectionText','roadCount','vehicleCount','vehicleInfo','phaseLight','phaseName','phaseDesc','cycle','time','created','passed','active','queue','run','pause','step1','step50','reset','rate','seed','mode','decisionStatus','trainActions','actionLvl','actionHdt','message'];
const elements=Object.fromEntries(ids.map(id=>[id,{id,textContent:'',innerHTML:'',value:id==='rate'?'5':id==='seed'?'1':id==='mode'?'fixed':'',disabled:false,className:'',style:{},listeners:{},addEventListener(type,fn){this.listeners[type]=fn},classList:{add(c){this.owner.className+=(this.owner.className?' ':'')+c},remove(c){this.owner.className=this.owner.className.split(/\s+/).filter(x=>x!==c).join(' ')},toggle(c,on){if(on)this.add(c);else this.remove(c)}}}]));
for(const e of Object.values(elements))e.classList.owner=e;
elements.map.getBoundingClientRect=()=>({width:800,height:500,left:0,top:0});elements.map.addEventListener=function(type,fn){this.listeners[type]=fn};
const ctx={setTransform(){},clearRect(){},beginPath(){},lineTo(){},moveTo(){},stroke(){},fill(){},save(){},restore(){},setLineDash(){},fillText(){},translate(){},rotate(){},fillRect(){},arc(){},roundRect(){}};
elements.map.getContext=()=>ctx;
const calls=[];let phase='none',cycle=0,mode='fixed',waiting=false,failNext=false;
const roads={type:'FeatureCollection',features:['NE','SE','SW','NW'].map((arm,i)=>({type:'Feature',geometry:{type:'LineString',coordinates:[[i*10,0],[i*10+5,5]]},properties:{id:arm+'_in',kind:'approach',arm}}))};
const snapshot=()=>({cycle,time_sec:cycle*.2,phase,mode,dqn_waiting_action:waiting,created:3,passed:1,active:1,queue:0,vehicles:[{id:1,road:'NE_in',progress:.4,lane:0,is_car:false}]});
async function fetchMock(path,options={}){let payload=options.body?JSON.parse(options.body):null;calls.push({path,method:options.method||'GET',payload});if(failNext){failNext=false;return{ok:false,status:503,json:async()=>({error:'simulated outage'})}}if(path==='/api/roads')return{ok:true,status:200,json:async()=>roads};if(path==='/api/state')return{ok:true,status:200,json:async()=>snapshot()};if(path==='/api/step'){if(mode==='train'&&waiting)return{ok:true,status:200,json:async()=>({...snapshot(),step_status:'waiting_for_action',cycles_advanced:0})};if(mode==='train'&&Math.floor((cycle+payload.cycles)/50)>Math.floor(cycle/50)){const boundary=(Math.floor(cycle/50)+1)*50,advanced=boundary-cycle;cycle=boundary;waiting=true;phase='none';return{ok:true,status:200,json:async()=>({...snapshot(),step_status:'waiting_for_action',cycles_advanced:advanced})}}cycle+=payload.cycles;phase=cycle%2?'LVL':'HDT';return{ok:true,status:200,json:async()=>({...snapshot(),step_status:'advanced',cycles_advanced:payload.cycles})}}if(path==='/api/action'){waiting=false;return{ok:true,status:200,json:async()=>snapshot()}}if(path==='/api/reset'){cycle=0;phase='none';mode=payload.mode;waiting=false;return{ok:true,status:200,json:async()=>snapshot()}}throw Error('unexpected '+path)}
const window={addEventListener(){}};
const sandbox={document:{getElementById:id=>elements[id]},window,devicePixelRatio:1,ResizeObserver:class{observe(){}},fetch:fetchMock,setTimeout,console,Map,Math,Number,String,Array,Promise};
vm.runInContext(source,vm.createContext(sandbox));
const flush=async()=>{for(let i=0;i<12;i++)await new Promise(r=>setImmediate(r))};
const click=async id=>{elements[id].listeners.click?.();await flush()};
(async()=>{
 await flush();assert.equal(elements.cycle.textContent,'0');assert(elements.phaseLight.className.includes('red'),'none phase badge should be red');
 await click('step1');assert.equal(calls.filter(c=>c.path==='/api/step').at(-1).payload.cycles,1);assert.equal(elements.cycle.textContent,'1');assert.equal(elements.phaseName.textContent,'LVL');
 await click('step50');assert.equal(calls.filter(c=>c.path==='/api/step').at(-1).payload.cycles,50);assert.equal(elements.cycle.textContent,'51');assert.equal(elements.vehicleCount.textContent,1);assert.equal(elements.created.textContent,'3');
 for(const rate of ['1','5','20']){elements.rate.value=rate;const before=calls.filter(c=>c.path==='/api/step').length;elements.run.listeners.click();await flush();assert.equal(elements.run.disabled,true);assert.equal(elements.pause.disabled,false);const auto=calls.filter(c=>c.path==='/api/step').slice(before);assert(auto.length>=1);assert.equal(auto[0].payload.cycles,Number(rate));elements.pause.listeners.click();await flush();assert.equal(elements.run.disabled,false);assert.equal(elements.pause.disabled,true);const stopped=calls.filter(c=>c.path==='/api/step').length;await new Promise(r=>setTimeout(r,210));assert.equal(calls.filter(c=>c.path==='/api/step').length,stopped,'pause must prevent another automatic request')}
 elements.seed.value='42';await click('reset');assert.equal(calls.filter(c=>c.path==='/api/reset').at(-1).payload.seed,42);assert.equal(elements.cycle.textContent,'0');
 elements.seed.value='1.5';const resetCount=calls.filter(c=>c.path==='/api/reset').length;await click('reset');assert.equal(calls.filter(c=>c.path==='/api/reset').length,resetCount);assert(elements.message.className.includes('show'),'invalid seed should be explained');
 elements.seed.value='7';failNext=true;await click('reset');assert(elements.message.textContent.includes('simulated outage'));assert(elements.connectionDot.className.includes('err'));
 failNext=true;await click('step1');assert(elements.message.textContent.includes('simulated outage'));
 elements.mode.value='train';elements.seed.value='42';await click('reset');const trainReset=calls.filter(c=>c.path==='/api/reset').at(-1);assert.equal(trainReset.payload.mode,'train');assert.equal(elements.mode.value,'train');
 await click('step50');assert.equal(calls.filter(c=>c.path==='/api/step').at(-1).payload.cycles,50);assert(elements.decisionStatus.className.includes('show'),'train decision wait must be visible');assert(elements.trainActions.className.includes('show'),'train actions must be available while waiting');
 const waitingSteps=calls.filter(c=>c.path==='/api/step').length;elements.run.listeners.click();await flush();await new Promise(r=>setTimeout(r,220));assert.equal(calls.filter(c=>c.path==='/api/step').length,waitingSteps,'automatic run must stop at pending decision');assert.equal(elements.run.disabled,true,'run button must be disabled while waiting');
 failNext=true;await click('actionLvl');assert(elements.message.textContent.includes('simulated outage'),'action API errors must be shown');assert(elements.trainActions.className.includes('show'),'failed action must preserve pending decision');
 await click('actionLvl');assert.equal(calls.filter(c=>c.path==='/api/action').at(-1).payload.action,0);assert(!elements.trainActions.className.includes('show'),'decision buttons hide after action');
 // Reach a second decision and verify the HDT action maps to integer 2.
 await click('step50');await click('actionHdt');assert.equal(calls.filter(c=>c.path==='/api/action').at(-1).payload.action,2);
 console.log('mocked DOM controls passed');
})().catch(e=>{console.error(e.stack||e);process.exitCode=1});'''
        result = subprocess.run(
            ["node", "-e", harness], input=self.html, text=True,
            capture_output=True, timeout=10, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
