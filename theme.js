
(function(){
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var observer = reduceMotion ? null : new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },{threshold:.12,rootMargin:"0px 0px -34px 0px"});

  function reveal(){
    var items=document.querySelectorAll(".hero > *, .main-grid > .card, .section-heading, #journal > .card, footer");
    items.forEach(function(el,index){
      if(el.dataset.motionReady==="1") return;
      el.dataset.motionReady="1";
      el.classList.add("motion-reveal");
      el.style.setProperty("--reveal-delay",Math.min(index*70,210)+"ms");
      if(reduceMotion) el.classList.add("is-visible"); else observer.observe(el);
    });
  }

  function tilt(){
    if(reduceMotion || !window.matchMedia("(pointer:fine)").matches) return;
    document.querySelectorAll(".card").forEach(function(card){
      if(card.dataset.tiltReady==="1") return;
      card.dataset.tiltReady="1";
      card.addEventListener("pointermove",function(e){
        var r=card.getBoundingClientRect();
        var px=(e.clientX-r.left)/r.width, py=(e.clientY-r.top)/r.height;
        card.style.setProperty("--tilt-x",((.5-py)*2).toFixed(2)+"deg");
        card.style.setProperty("--tilt-y",((px-.5)*2.4).toFixed(2)+"deg");
        card.style.setProperty("--card-x",(px*100).toFixed(1)+"%");
        card.style.setProperty("--card-y",(py*100).toFixed(1)+"%");
      });
      card.addEventListener("pointerleave",function(){
        card.style.setProperty("--tilt-x","0deg");
        card.style.setProperty("--tilt-y","0deg");
      });
    });
  }

  function ripple(){
    document.addEventListener("pointerdown",function(e){
      var button=e.target.closest("button");
      if(!button || button.disabled || reduceMotion) return;
      var r=button.getBoundingClientRect();
      var span=document.createElement("span");
      span.className="ui-ripple";
      span.style.left=(e.clientX-r.left)+"px";
      span.style.top=(e.clientY-r.top)+"px";
      button.appendChild(span);
      setTimeout(function(){span.remove()},680);
    },{passive:true});
  }

  function ambient(){
    if(reduceMotion || !window.matchMedia("(pointer:fine)").matches) return;
    var ticking=false,x=0,y=0;
    window.addEventListener("pointermove",function(e){
      x=e.clientX;y=e.clientY;
      if(ticking) return;
      ticking=true;
      requestAnimationFrame(function(){
        document.documentElement.style.setProperty("--spot-x",Math.max(0,Math.min(100,x/window.innerWidth*100)).toFixed(1)+"%");
        document.documentElement.style.setProperty("--spot-y",Math.max(0,Math.min(100,y/window.innerHeight*100)).toFixed(1)+"%");
        ticking=false;
      });
    },{passive:true});
  }

  window.addEventListener("DOMContentLoaded",function(){
    reveal();tilt();ripple();ambient();
  });
})();
