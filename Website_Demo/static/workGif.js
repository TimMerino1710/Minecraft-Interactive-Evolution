self.importScripts( "voxworldjs/three.js" );
self.importScripts( "voxworldjs/b64.js" );
self.importScripts( "voxworldjs/GIFEncoder.js" );
self.importScripts( "voxworldjs/LZWEncoder.js" );
self.importScripts( "voxworldjs/NeuQuant.js" );
self.importScripts( "voxworldjs/struct2data.js" );

function makeGIF(in_dat){
    //re-render the images
    var arr3d = JSON.parse(in_dat.houses);
    let TEXTURE_DAT = JSON.parse(in_dat.textures);
    let out_dat = renderGIFdat(arr3d,TEXTURE_DAT,{angle:230,radius:18,height:"center"});
    postMessage(out_dat)
}
onmessage = function(e) {
    makeGIF(e.data);
};