use std::path::Path;
use tensorflow::eager::{raw_ops, Context, ContextOptions};
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;
use tensorflow::{Code, DataType, Result, Status, Tensor};

use once_cell::sync::Lazy;

// Generate global execution context
static CONTEXT: Lazy<Context> = Lazy::new(|| {
    let opts = ContextOptions::new();
    Context::new(opts).unwrap()
});

fn read_image_224x224(fname: &str, dump_file: bool) -> Result<Tensor<f32>> {
    // read sample image
    let filename = Tensor::from(String::from(fname));
    let contents = raw_ops::read_file(&CONTEXT, filename)?;
    let img = raw_ops::decode_jpeg(&CONTEXT, contents)?;

    // resize (TF2 ver) - use `scale_and_translate` to support antialiasing option
    let height = img.dim(0)?;
    let width = img.dim(1)?;
    dbg!(width);
    dbg!(height);

    // expand 3-d to 4-d tensor
    let images = raw_ops::expand_dims(&CONTEXT, img, Tensor::from(&[0]))?;
    let size = Tensor::from([224, 224]);
    let cast_to_float = raw_ops::Cast::new().DstT(DataType::Float);

    // calculate scaling factor
    let scale = raw_ops::div(
        &CONTEXT,
        cast_to_float.call(&CONTEXT, size.clone())?,
        Tensor::from([height as f32, width as f32]),
    )?;

    // check scaling factor
    let scale_tensor: Tensor<f32> = scale.copy_sharing_tensor()?.resolve()?;
    dbg!((scale_tensor[0], scale_tensor[1]));

    let translation = Tensor::from([0f32, 0f32]); // no translation

    // execute resize
    let smalls = raw_ops::ScaleAndTranslate::new().antialias(true).call(
        &CONTEXT,
        images,
        size,
        scale,
        translation,
    )?;

    // dump result
    let image: Tensor<f32> = smalls.copy_sharing_tensor()?.resolve()?;
    if dump_file {
        let img = raw_ops::squeeze(&CONTEXT, smalls)?;
        let cast_to_uint8 = raw_ops::Cast::new().DstT(DataType::UInt8);
        let img = cast_to_uint8.call(&CONTEXT, img)?;
        let contents = raw_ops::encode_png(&CONTEXT, img)?;
        let filename = Tensor::from(String::from("sample_images/small.png"));
        raw_ops::write_file(&CONTEXT, filename, contents)?;
    }

    Ok(image)
}

fn main() -> Result<()> {
    let fname = "sample_images/macaque.jpg";
    let img = read_image_224x224(fname, false)?;

    let model_dir = "model";
    if !Path::new(model_dir).is_dir() {
        println!(
            "\"{}\" not found. Please generate model files with `python python/create_model.py`",
            model_dir
        );
        let mut status = Status::new();
        status.set(
            Code::NotFound,
            &format!("\"{}\" directory not found", model_dir),
        )?;
        return Err(status);
    }

    // Load the saved model exported by python/create_model.py.
    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_dir)?;
    let session = &bundle.session;

    // get in/out operations
    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    let x_info = signature.get_input("input_1")?;
    let op_x = &graph.operation_by_name_required(&x_info.name().name)?;
    let output_info = signature.get_output("Predictions")?;
    let op_output = &graph.operation_by_name_required(&output_info.name().name)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(op_x, 0, &img);
    let token_output = args.request_fetch(op_output, 0);
    session.run(&mut args)?;

    // Check the output.
    let output: Tensor<f32> = args.fetch(token_output)?;

    // Get arg_max index. This is expected to be the same as the one from the Python code
    let idx: Tensor<i64> = raw_ops::arg_max(&CONTEXT, output, Tensor::from(-1))?.resolve()?;
    dbg!(&idx);
    dbg!(idx[0]);

    Ok(())
}
