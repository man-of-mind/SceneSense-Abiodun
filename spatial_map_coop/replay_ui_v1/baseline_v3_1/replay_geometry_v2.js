/* World-metre 2D building approximation; never qualified sensor visibility. */
(function(root){'use strict';
const cross=(a,b)=>a[0]*b[1]-a[1]*b[0], sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
function hit(o,d,a,b){const v=sub(b,a),q=sub(a,o),den=cross(d,v);if(Math.abs(den)<1e-9)return null;
 const t=cross(q,v)/den,u=cross(q,d)/den;return t>=0&&u>=0&&u<=1?{t,u}:null;}
function inside(p,poly){let v=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){
 const a=poly[i],b=poly[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])v=!v;}return v;}
class Buildings {
 constructor(data){this.valid=!!(data?.validation?.valid&&data.buildings?.length);this.polys=data?.buildings||[];this.edges=[];this.grid=new Map();this.cell=20;
  if(!this.valid)return;
  this.polys.forEach((poly,id)=>poly.forEach((a,i)=>{const b=poly[(i+1)%poly.length],edge={a,b,id,index:this.edges.length};this.edges.push(edge);
   for(let x=Math.floor(Math.min(a[0],b[0])/this.cell);x<=Math.floor(Math.max(a[0],b[0])/this.cell);x++)
    for(let y=Math.floor(Math.min(a[1],b[1])/this.cell);y<=Math.floor(Math.max(a[1],b[1])/this.cell);y++){
     const k=x+','+y;if(!this.grid.has(k))this.grid.set(k,[]);this.grid.get(k).push(edge);}}));
 }
 nearby(o,r){const found=new Map();for(let x=Math.floor((o[0]-r)/this.cell);x<=Math.floor((o[0]+r)/this.cell);x++)
  for(let y=Math.floor((o[1]-r)/this.cell);y<=Math.floor((o[1]+r)/this.cell);y++)for(const e of this.grid.get(x+','+y)||[])found.set(e.index,e);return [...found.values()];}
 ray(o,d,range){let nearest=range;const seen=new Set();
  // Grid traversal samples at half a cell, including all four neighbouring
  // cells; intersection remains exact, so sampling never sets ray distance.
  for(let t=0;t<=range+this.cell/2;t+=this.cell/2){const x=Math.floor((o[0]+d[0]*Math.min(t,range))/this.cell),y=Math.floor((o[1]+d[1]*Math.min(t,range))/this.cell);
   for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const e of this.grid.get((x+dx)+','+(y+dy))||[]){
    if(seen.has(e.index))continue;seen.add(e.index);const h=hit(o,d,e.a,e.b);if(h&&h.t<nearest)nearest=h.t;}
   if(t>nearest+this.cell)break;
  }return nearest;
 }
 polygon(sensor,fov,range){const key=[sensor.x,sensor.y,sensor.yaw,fov,range].join(',');if(this.cacheKey===key)return this.cache;this.cacheKey=key;this.cache=this.computePolygon(sensor,fov,range);return this.cache;}
 computePolygon(sensor,fov,range){if(!this.valid)return null;const o=[sensor.x,sensor.y];
  const nearby=this.nearby(o,range),ids=new Set(nearby.map(e=>e.id));if([...ids].some(id=>inside(o,this.polys[id])))return null;
  const yaw=sensor.yaw*Math.PI/180,half=fov*Math.PI/360,angles=[-half,0,half];
  for(let a=-half;a<half;a+=Math.PI/90)angles.push(a);
  for(const e of nearby)for(const p of [e.a,e.b]){const a=Math.atan2(p[1]-o[1],p[0]-o[0])-yaw,wrapped=Math.atan2(Math.sin(a),Math.cos(a));
   for(const delta of [-1e-6,0,1e-6])if(wrapped+delta>-half&&wrapped+delta<half)angles.push(wrapped+delta);}
  angles.sort((a,b)=>a-b);return [o,...angles.filter((a,i)=>i===0||a-angles[i-1]>1e-8).map(a=>{
   const d=[Math.cos(a+yaw),Math.sin(a+yaw)],r=this.ray(o,d,range);return [o[0]+d[0]*r,o[1]+d[1]*r];})];
 }
 classify(sensor,pose,dimensions,yaw){if(!this.valid||!sensor)return {state:'VISIBILITY_UNCERTAIN',reason:'MISSING_VALIDATED_STATIC_GEOMETRY'};
  const o=[sensor.x,sensor.y],l=Number(dimensions?.length),w=Number(dimensions?.width);
  if(!(l>0&&w>0&&l<30&&w<12))return {state:'VISIBILITY_UNCERTAIN',reason:'INVALID_OBJECT_FOOTPRINT'};
  // Expand every footprint by 0.3 m. Unknown heading gets a circumscribed box.
  const known=Number.isFinite(yaw),a=known?yaw*Math.PI/180:0,h=known?l/2+.3:Math.hypot(l,w)/2+.3,k=known?w/2+.3:h;
  const corners=[[-h,-k],[h,-k],[h,k],[-h,k]].map(([x,y])=>[pose.x+x*Math.cos(a)-y*Math.sin(a),pose.y+x*Math.sin(a)+y*Math.cos(a)]);
  const range=Math.hypot(pose.x-o[0],pose.y-o[1])+Math.hypot(h,k),edges=this.nearby(o,range);
  const ids=new Set(edges.map(e=>e.id));if([...ids].some(id=>inside(o,this.polys[id])||corners.some(p=>inside(p,this.polys[id]))))return {state:'VISIBILITY_UNCERTAIN',reason:'SENSOR_OR_FOOTPRINT_INTERSECTS_BUILDING'};
  let any=false;
  for(const e of edges){let complete=true;const len=Math.hypot(e.b[0]-e.a[0],e.b[1]-e.a[1]);
   for(const p of corners){const d=sub(p,o),dist=Math.hypot(...d),v=d.map(x=>x/dist),q=hit(o,v,e.a,e.b);
    if(q&&q.t<dist)any=true;
    // One boundary segment must cover the ENTIRE expanded convex footprint,
    // with 0.5 m depth and 0.25 m corner margins. Mixed/corner cases stay drawn.
    if(!q||q.t>=dist-.5||q.u*len<.25||(1-q.u)*len<.25)complete=false;}
   if(complete)return {state:'STATIC_MAP_OCCLUDED',reason:'ENTIRE_EXPANDED_FOOTPRINT_BEHIND_ONE_BOUNDARY',building:e.id};
  }
  return {state:any?'VISIBILITY_UNCERTAIN':'NO_MODELED_OBSTRUCTION',reason:any?'PARTIAL_OR_CORNER_EXPOSURE':'2D_BUILDINGS_ONLY'};
 }
}
const api={Buildings,hit,inside};root.ScenesenseBuildings=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
