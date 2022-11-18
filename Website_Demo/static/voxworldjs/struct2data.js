// IMPORTS A JSON STRING, PARSES IT, RENDERS THE STRUCTURES, AND RETURNS THE IMAGE DATA
// Written by Milk

//////////////      VARIABLES      //////////////

var CONFIG = {
    BG_COLOR : 0xfafafa,
    CANV_WIDTH : 400,
    CANV_HEIGHT : 300,

    RADIUS : 15,
    ANGLE : 0,
    CENTER_Y : 2,
    DEF_ANGLE : 180,
    DEF_RADIUS : 15,
    DEF_CENTER_Y : 2,
    use_struct_center : true,

    CUR_TEXTURE_LIST : ["air","stonebrick","dirt","planks_oak","sand","iron_bars","glass","iron_block","log_oak","wool_colored_red","stone_slab_side"]

}

let RAW_TEXT_DAT = {};
let TEXTURE_PNG = {};
let TEXT_MAT = {};



//////////////      THREE.JS SETUP     ///////////////

//set up the canvas
var rendCanvas = null;
if (typeof(document) !== "undefined"){
    rendCanvas = (document.createElement('canvas'));
    rendCanvas.id = "rendCanvas";
    rendCanvas.width = CONFIG.CANV_WIDTH;
    rendCanvas.height = CONFIG.CANV_HEIGHT;
    document.getElementById("container").appendChild(rendCanvas);
}else{
    rendCanvas = new OffscreenCanvas(CONFIG.CANV_WIDTH,CONFIG.CANV_HEIGHT);
    rendCanvas.style = {width: CONFIG.CANV_WIDTH, height: CONFIG.CANV_HEIGHT};
}

//set up the renderer
const RENDERER = new THREE.WebGLRenderer({canvas: rendCanvas, antialias: false, preserveDrawingBuffer: true});
RENDERER.setSize(400, 300);
// RENDERER.domElement.id = "renderCanvas";
// document.getElementById("render").appendChild(RENDERER.domElement);

//setup scene and camera
const SCENE = new THREE.Scene();
SCENE.background = new THREE.Color( CONFIG.BG_COLOR ).convertSRGBToLinear();
var CAMERA = null;
if(typeof(document) !== "undefined"){
    CAMERA = new THREE.PerspectiveCamera( 75, rendCanvas.clientWidth / rendCanvas.clientHeight, 0.1, 1000);
}else{
    CAMERA = new THREE.PerspectiveCamera( 75, 400 / 300, 0.1, 1000);
}


//load a texture loader
var loader = new THREE.TextureLoader(  );
loader.crossOrigin = true;

//add a light
const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
SCENE.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
directionalLight.position.set(10, 20, 0); // x, y, z
SCENE.add(directionalLight);

//set cube geometry to use as voxels
let VOXEL = new THREE.BoxGeometry( 1, 1, 1 );


//////////////      RENDER FUNCTIONS     ///////////////


//import the list of textures from the JSON file (textures.json)
function importAllTextures(){
    // fetch('./textures/textures.json')
    fetch('static/voxworldjs/textures/textures.json')
    .then((response) => response.json())
    .then((json) =>  {
        ALL_TEXTURES = json.textures;

        //import the images and add them as materials
        for(let t=0;t<ALL_TEXTURES.length;t++){
            let image = new Image();
            // image.src = "./textures/"+ALL_TEXTURES[t]+".png";
            image.src = "static/voxworldjs/textures/"+ALL_TEXTURES[t]+".png"

            image.onload = function(){
                loadTextureIMG(image,ALL_TEXTURES[t]);
            }
        }
    });
}

//only import the textures in the CUR_TEXTURE_LIST
function importSubTextures(){
    for(let t=0;t<CONFIG.CUR_TEXTURE_LIST.length;t++){
        //skip air (not rendered)
        if(CONFIG.CUR_TEXTURE_LIST[t] == "air")
            continue;

        let image = new Image();
        // image.src = "./textures/"+ALL_TEXTURES[t]+".png";
        image.src = "static/voxworldjs/textures/"+CONFIG.CUR_TEXTURE_LIST[t]+".png"

        image.onload = function(){
            loadTextureIMG(image,CONFIG.CUR_TEXTURE_LIST[t]);
            console.log("loaded texture: "+CONFIG.CUR_TEXTURE_LIST[t]);
        }
    }
}

//import the textures by rendering their images onto a fake canvas and saving as a material
function loadTextureIMG(image,id){
    //fake mini canvas for the imported images
    const img_canvas = document.createElement("canvas")
    img_canvas.width = image.width
    img_canvas.height = image.height
    const itx = img_canvas.getContext('2d')

    //draw onto mini canvas
    itx.clearRect(0, 0, img_canvas.width, img_canvas.height)
    itx.drawImage(image, 0, 0, img_canvas.width, img_canvas.height);

    //add to the scene
    RAW_TEXT_DAT[id] = Array.from(itx.getImageData(0, 0, image.width, image.height).data);
    TEXTURE_PNG[id] = new THREE.CanvasTexture(img_canvas);
    TEXT_MAT[id] = new THREE.MeshBasicMaterial({map: TEXTURE_PNG[id],transparent: true});
}

//i hate this function
function getTextureData(){
    return {"names":CONFIG.CUR_TEXTURE_LIST,"data":RAW_TEXT_DAT};
}

//i hate this function even more
function reimportTextures(raw_imgs){
    //assume you only call this for the webworker
    let keys = Object.keys(raw_imgs.data);
    for(let i=0;i<keys.length;i++){
        //make a new image data object from the array passed from the data file
        let imdat = new ImageData(16,16);
        imdat.data.set(raw_imgs.data[keys[i]]);

        //create a new texture
        TEXTURE_PNG[keys[i]] = new THREE.Texture(imdat);
        TEXTURE_PNG[keys[i]].needsUpdate = true; ///CRUCIAL! OR WILL NOT SHOW UP
        TEXT_MAT[keys[i]] = new THREE.MeshBasicMaterial({map: TEXTURE_PNG[keys[i]],transparent: true});
    }
}

//remove all structures from the scene
function clearScene(){
    //remove any previous structures
    while(SCENE.children.length > 0){ 
        SCENE.remove(SCENE.children[0]); 
    }
}

//make the structure given a 3d array
//represented as x,y,z in the 3d array
function make3dStructure(arr3d,offset=[0,0,0]){
    //check if a structure was passed
    if(arr3d == null || arr3d.length == 0){
        alert("No structure to render!\nPlease import a structure or paste one in the textarea to the right.");
        return;
    }

    //remove any previous structures
    clearScene();

    //default offset
    let def_off = [0.5,0.5,0.5];
    let off = [offset[0]+def_off[0],offset[1]+def_off[1],offset[2]+def_off[2]];

    //get the structure properties
    let structDim = [arr3d.length, arr3d[0].length, arr3d[0][0].length]; //w,h,d
    let structCen = [structDim[0]/2, structDim[1]/2, structDim[2]/2]; //x,y,z

    //build the structure
    let structObj = new THREE.Group();
    for(let i = 0; i < arr3d.length; i++){
        for(let j = 0; j < arr3d[i].length; j++){
            for(let k = 0; k < arr3d[i][j].length; k++){
                if(CONFIG.CUR_TEXTURE_LIST[arr3d[i][j][k]] != "air"){
                    let cube = new THREE.Mesh( VOXEL, TEXT_MAT[CONFIG.CUR_TEXTURE_LIST[arr3d[i][j][k]]] );
                    cube.position.set(i+off[0]-structCen[0],structDim[1]-j+off[1],k+off[2]-structCen[2]);
                    structObj.add(cube);
                }
            }
        }
    }

    //move camera to the center of the structure
    if(CONFIG.use_struct_center)
        CONFIG.CENTER_Y = Math.max(2,structCen[1]);

    //move the camera into position
    console.log(`> Camera position set to: angle=${CONFIG.ANGLE}, zoom=${CONFIG.RADIUS}, height=${CONFIG.CENTER_Y}`);
    rotateCam(CONFIG.ANGLE,CONFIG.CENTER_Y,CONFIG.RADIUS);

    //add the structure to the scene
    SCENE.add(structObj);

    //render the scene
    RENDERER.render(SCENE, CAMERA);

}

//reset the camera's angle and position
function resetCamera(){
    CONFIG.ANGLE = CONFIG.DEF_ANGLE;
    CONFIG.RADIUS = CONFIG.DEF_RADIUS;
    CONFIG.CENTER_Y = CONFIG.DEF_CENTER_Y;

    rotateCam(CONFIG.ANGLE,CONFIG.CENTER_Y,CONFIG.RADIUS)
}

//rotate the camera around the structure
function rotateCam(angle,height,radius){
    CAMERA.position.y = height;
    CAMERA.position.x = radius * Math.cos( angle * (Math.PI/180) );  
    CAMERA.position.z = radius * Math.sin( angle * (Math.PI/180) ); 
    CAMERA.lookAt(0,height,0);
}

/////////////   EXPORTING FUNCTIONS  ///////////////



//return the data of the PNG rendered image
function renderPNGdat(arr3d){
    //pass in the structure
    make3dStructure(arr3d);

    //load the image from the renderer canvas
    let im_data = RENDERER.domElement.toDataURL("image/png");
    return im_data;
}

//return the data of the GIF rendered image
function renderGIFdat(arr3d,text_dat=null,cam=null){
    if(cam != null){
        CONFIG.ANGLE = cam.angle;
        CONFIG.RADIUS = cam.radius;
        CONFIG.CENTER_Y = cam.height;
    }
    //fix the textures
    if(text_dat != null){
        reimportTextures(text_dat);
    }

    //pass in the structure
    make3dStructure(arr3d);

    //make a fake canvas to copy from the renderer
    let fake_canvas = null;
    if(typeof(document) !== "undefined"){
        fake_canvas = document.createElement("canvas");
        fake_canvas.width = RENDERER.domElement.width;
        fake_canvas.height = RENDERER.domElement.height;
    }else{
        fake_canvas = new OffscreenCanvas(CONFIG.CANV_WIDTH, CONFIG.CANV_HEIGHT);
    }
    let fake_ctx = fake_canvas.getContext("2d",{willReadFrequently: true});


    //make the encoding and start 
    var encoder = new GIFEncoder();
    encoder.setRepeat(0);  //0  -> loop forever
    encoder.setDelay(100);   //1  -> frame delay in ms
    encoder.setSize( CONFIG.CANV_WIDTH, CONFIG.CANV_HEIGHT );
    encoder.start();

    //repeat for a full rotation
    let FRAMES = 36;
    for(let i=0;i<FRAMES;i++){
        //rotate, render, and copy
        let newAngle = CONFIG.ANGLE + (360/FRAMES)*i;
        rotateCam(newAngle%360,CONFIG.CENTER_Y,CONFIG.RADIUS);
        RENDERER.render(SCENE, CAMERA);
        fake_ctx.drawImage(RENDERER.domElement, 0, 0);

        //add the frame to the encoder
        encoder.addFrame(fake_ctx);
        // console.log(`> Frame ${i+1}/${FRAMES} added to GIF`);
    }

    //finish the encoding and save to the image
    encoder.finish();
    return "data:image/gif;base64,"+encode64(encoder.stream().getData());
}   


/////////////////    MAIN CALLS   //////////////////

//initializing function for the structures
function init_rend(){
    console.log("struct2data init");
    // importTextures();
    importSubTextures();
    resetCamera();
    rendCanvas.style.display = "none";
}
