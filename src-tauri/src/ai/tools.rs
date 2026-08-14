// src-tauri/src/ai/tools.rs
//! The vocabulary the language router can call.
//!
//! This is the contract between "what the user said" and "what Nabu does". It
//! is deliberately declared once, in data: the schema handed to the model and
//! the actions the interface can perform must never drift apart, because a
//! model that calls a tool the frontend does not implement fails silently.
//!
//! Kept free of any dependency on the language engine so the list stays
//! available — and testable — even when the `llm` feature is off.

use serde::Serialize;

/// One parameter of a tool, in the subset of JSON Schema the models understand.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ToolParam {
    pub name: &'static str,
    /// JSON Schema type: `number`, `string` or `boolean`.
    pub kind: &'static str,
    pub description: &'static str,
    pub required: bool,
    /// Bounds for numeric parameters, mirrored into the schema as a hint.
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub params: &'static [ToolParam],
}

const fn amount(description: &'static str) -> ToolParam {
    ToolParam {
        name: "amount",
        kind: "number",
        description,
        required: true,
        min: Some(-100.0),
        max: Some(100.0),
    }
}

/// Everything the router may call.
///
/// Descriptions are written for the model, not for the interface: they say when
/// to pick a tool and how the number behaves, because that is what the model
/// has to reason about.
pub const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "remove_background",
        description: "Remove the background of the photo, leaving the main subject on transparency. Use when the user wants to isolate the subject or delete everything behind it.",
        params: &[],
    },
    ToolSpec {
        name: "cut_out_selection",
        description: "Cut out the object the user has selected, isolating it from the rest of the photo. Only valid when a selection exists.",
        params: &[],
    },
    ToolSpec {
        name: "remove_selection",
        description: "Erase whatever is inside the user's current selection and rebuild the background behind it. Only valid when a selection exists. Prefer it over remove_object when the user says to remove 'this' or the thing they selected, and when what should be removed is not in the SCENE list but a selection exists.",
        params: &[],
    },
    ToolSpec {
        name: "select_object",
        description: "Select one object from the SCENE list by its id number. Use when the user points at a thing in the photo — the person, the red car, the dog on the left. Pick the id whose class, colour and position best match the request. Only valid when the context contains a SCENE list.",
        params: &[ToolParam {
            name: "id",
            kind: "number",
            description: "Id of the object in the SCENE list: 2 selects the entry '#2'.",
            required: true,
            min: Some(1.0),
            max: None,
        }],
    },
    ToolSpec {
        name: "remove_object",
        description: "Erase one object from the photo and rebuild the background behind it. Use when the user asks to remove, delete or get rid of a thing that appears in the SCENE list; pick the id whose class, colour and position best match. If nothing in the SCENE matches what should be removed, use `report_unsupported` with feature `remove-object` instead of guessing.",
        params: &[ToolParam {
            name: "id",
            kind: "number",
            description: "Id of the object in the SCENE list: 2 removes the entry '#2'.",
            required: true,
            min: Some(1.0),
            max: None,
        }],
    },
    ToolSpec {
        name: "upscale",
        // Deliberately contrasted with `resize`: both make a photo bigger, and
        // a model given two overlapping descriptions will pick either at random.
        description: "Double the resolution using a model that reconstructs detail, making a small or soft photo genuinely sharper. Use for requests about quality, sharpness or rescuing a low-resolution photo. For a plain change of dimensions, use `resize` instead.",
        params: &[],
    },
    ToolSpec {
        name: "adjust",
        description: "Change one tonal or colour property of the photo. Positive amounts increase the property, negative amounts decrease it; 0 means no change. Use this for brightness, colour and detail requests.",
        params: &[
            ToolParam {
                name: "property",
                kind: "string",
                // Each name is glossed, because the model has no idea what a
                // photographer means by "temperature" otherwise: asked to fix
                // colours that look "too cold" it reached for saturation.
                description: "Which property to change. \
exposure = overall brightness, darker or lighter. \
contrast = difference between light and dark. \
highlights = only the bright areas. shadows = only the dark areas. \
whites = the brightest point. blacks = the darkest point. \
temperature = white balance, positive is warmer/more orange, negative is colder/more blue. \
tint = green versus magenta cast. \
vibrance = strength of muted colours only. saturation = strength of all colours. \
hue = shifts every colour around the wheel. \
clarity = local contrast, midtone punch. sharpness = edge definition. blur = softens. \
vignette = darkens the corners. grain = adds film noise.",
                required: true,
                min: None,
                max: None,
            },
            amount("How much to change it, from -100 to 100. A plain request like 'brighter' is about 25; 'much brighter' is about 60."),
        ],
    },
    ToolSpec {
        name: "resize",
        description: "Resample the photo to different pixel dimensions. Use for requests about size in pixels or a percentage. This does NOT add detail: to make a small photo genuinely sharper, use `upscale` instead.",
        params: &[
            ToolParam {
                name: "percent",
                kind: "number",
                description: "Resize to this percentage of the current size. 50 halves it, 200 doubles it. Give this OR width/height, not both.",
                required: false,
                min: Some(1.0),
                max: Some(1000.0),
            },
            ToolParam {
                name: "width",
                kind: "number",
                description: "Target width in pixels. If only this is given, the height follows the proportions.",
                required: false,
                min: Some(16.0),
                max: Some(20000.0),
            },
            ToolParam {
                name: "height",
                kind: "number",
                description: "Target height in pixels. If only this is given, the width follows the proportions.",
                required: false,
                min: Some(16.0),
                max: Some(20000.0),
            },
        ],
    },
    ToolSpec {
        name: "reset_adjustments",
        description: "Undo every tonal and colour adjustment, returning the photo to how it looked when opened.",
        params: &[],
    },
    ToolSpec {
        // A small model forced to always call a tool will otherwise shoehorn
        // an impossible request into the nearest tool and silently do the
        // wrong thing. Giving it an honest "cannot do that yet" keeps the
        // answer truthful and tells the interface which message to show.
        name: "report_unsupported",
        description: "Report that the request is understood but needs a capability Nabu does not have yet. Use for replacing or adding content, generating imagery, or removing something that does NOT appear in the SCENE list. Selecting and removing objects that ARE in the SCENE list are supported, via `select_object` and `remove_object`.",
        params: &[ToolParam {
            name: "feature",
            kind: "string",
            description: "Which missing capability the request needs. remove-object = erase something from the photo and fill the hole. replace-content = swap or add content (sky, faces, objects). generate = create new imagery from scratch.",
            required: true,
            min: None,
            max: None,
        }],
    },
];

/// Values `report_unsupported.feature` accepts, mirrored into the schema.
pub const UNSUPPORTED_FEATURES: &[&str] = &["remove-object", "replace-content", "generate"];

/// Properties `adjust` accepts, matching the `Adjustments` record on the
/// frontend. Declared here so a typo in a model's answer can be rejected.
pub const ADJUSTMENT_PROPERTIES: &[&str] = &[
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
    "hue",
    "clarity",
    "sharpness",
    "blur",
    "vignette",
    "grain",
];

/// What the router decided, ready to be sent to the interface.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub name: String,
    /// Arguments as a JSON object, validated against the spec above.
    pub arguments: serde_json::Value,
}

/// Words the model reaches for that mean one of our properties.
///
/// A small model answers with the everyday term rather than the one in the
/// schema. Rejecting "brightness" because the slider is called "exposure"
/// would be pedantry the user pays for.
const PROPERTY_SYNONYMS: &[(&str, &str)] = &[
    ("brightness", "exposure"),
    ("bright", "exposure"),
    ("light", "exposure"),
    ("luminosity", "exposure"),
    ("luminosita", "exposure"),
    ("warmth", "temperature"),
    ("warm", "temperature"),
    ("colour_temperature", "temperature"),
    ("color_temperature", "temperature"),
    ("saturation_level", "saturation"),
    ("colour", "saturation"),
    ("color", "saturation"),
    ("vividness", "vibrance"),
    ("sharpen", "sharpness"),
    ("detail", "sharpness"),
    ("softness", "blur"),
    ("noise", "grain"),
    ("shadow", "shadows"),
    ("highlight", "highlights"),
    ("white", "whites"),
    ("black", "blacks"),
];

/// Maps a property the model chose onto one Nabu actually has.
pub fn canonical_property(name: &str) -> Option<&'static str> {
    let lowered = name.trim().to_ascii_lowercase();

    if let Some(exact) = ADJUSTMENT_PROPERTIES.iter().find(|p| **p == lowered) {
        return Some(exact);
    }
    PROPERTY_SYNONYMS
        .iter()
        .find(|(from, _)| *from == lowered)
        .map(|(_, to)| *to)
}

pub fn find(name: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|t| t.name == name)
}

/// Builds the JSON Schema for a tool's parameters.
pub fn parameters_schema(spec: &ToolSpec) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for param in spec.params {
        let mut entry = serde_json::Map::new();
        entry.insert("type".into(), serde_json::json!(param.kind));
        entry.insert("description".into(), serde_json::json!(param.description));
        if let Some(min) = param.min {
            entry.insert("minimum".into(), serde_json::json!(min));
        }
        if let Some(max) = param.max {
            entry.insert("maximum".into(), serde_json::json!(max));
        }
        // Listing the accepted values as an enum steers the model far better
        // than describing them in prose: left to itself it invents reasonable
        // synonyms like "brightness" that the interface cannot act on.
        if spec.name == "adjust" && param.name == "property" {
            entry.insert("enum".into(), serde_json::json!(ADJUSTMENT_PROPERTIES));
        }
        if spec.name == "report_unsupported" && param.name == "feature" {
            entry.insert("enum".into(), serde_json::json!(UNSUPPORTED_FEATURES));
        }
        properties.insert(param.name.into(), serde_json::Value::Object(entry));
        if param.required {
            required.push(param.name);
        }
    }

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

/// Rejects a call the interface could not carry out.
///
/// A small model will occasionally invent a tool or a property. Catching it
/// here means the user sees "I did not understand" instead of the application
/// silently doing nothing.
pub fn validate(call: &ToolCall) -> Result<(), String> {
    let spec = find(&call.name).ok_or_else(|| format!("unknown tool: {}", call.name))?;

    for param in spec.params.iter().filter(|p| p.required) {
        let value = call
            .arguments
            .get(param.name)
            .ok_or_else(|| format!("missing argument: {}", param.name))?;

        match param.kind {
            "number" if !value.is_number() => {
                return Err(format!("{} must be a number", param.name));
            }
            "string" if !value.is_string() => {
                return Err(format!("{} must be a string", param.name));
            }
            _ => {}
        }
    }

    Ok(())
}

/// Rewrites a call into the exact shape the interface expects.
///
/// Returns an error only when nothing sensible can be salvaged, so a near-miss
/// like "brightness" becomes "exposure" rather than a failure the user sees.
pub fn normalise(call: &mut ToolCall) -> Result<(), String> {
    validate(call)?;

    if call.name == "adjust" {
        let raw = call
            .arguments
            .get("property")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let canonical =
            canonical_property(&raw).ok_or_else(|| format!("unknown property: {raw}"))?;

        if let Some(object) = call.arguments.as_object_mut() {
            object.insert("property".into(), serde_json::json!(canonical));
        }
    }

    if call.name == "report_unsupported" {
        // The interface shows the same message whichever capability is
        // missing, so an invented value is folded into the generic one rather
        // than turned into an error the user has to read.
        let raw = call
            .arguments
            .get("feature")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        if !UNSUPPORTED_FEATURES.contains(&raw.as_str()) {
            if let Some(object) = call.arguments.as_object_mut() {
                object.insert("feature".into(), serde_json::json!("generate"));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_a_description_and_unique_name() {
        let mut names: Vec<&str> = TOOLS.iter().map(|t| t.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate tool names");

        for tool in TOOLS {
            assert!(!tool.description.is_empty(), "{} has no description", tool.name);
            for param in tool.params {
                assert!(!param.description.is_empty(), "{}.{} has no description", tool.name, param.name);
            }
        }
    }

    #[test]
    fn schema_marks_required_parameters() {
        let spec = find("adjust").expect("adjust exists");
        let schema = parameters_schema(spec);

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["property"].is_object());
        assert_eq!(schema["properties"]["amount"]["minimum"], -100.0);

        let required = schema["required"].as_array().expect("required list");
        assert!(required.iter().any(|v| v == "property"));
        assert!(required.iter().any(|v| v == "amount"));
    }

    #[test]
    fn maps_everyday_words_onto_the_real_property() {
        assert_eq!(canonical_property("brightness"), Some("exposure"));
        assert_eq!(canonical_property("Warmth"), Some("temperature"));
        assert_eq!(canonical_property("exposure"), Some("exposure"));
        assert_eq!(canonical_property("sparkle"), None);
    }

    #[test]
    fn normalise_rewrites_a_synonym_in_place() {
        let mut call = ToolCall {
            name: "adjust".into(),
            arguments: serde_json::json!({ "property": "brightness", "amount": 25 }),
        };
        normalise(&mut call).expect("normalised");
        assert_eq!(call.arguments["property"], "exposure");
    }

    #[test]
    fn the_schema_constrains_the_property_to_a_list() {
        let schema = parameters_schema(find("adjust").expect("adjust"));
        let values = schema["properties"]["property"]["enum"]
            .as_array()
            .expect("enum present");
        assert!(values.iter().any(|v| v == "exposure"));
        assert_eq!(values.len(), ADJUSTMENT_PROPERTIES.len());
    }

    #[test]
    fn rejects_invented_tools_and_properties() {
        let unknown = ToolCall {
            name: "make_it_pretty".into(),
            arguments: serde_json::json!({}),
        };
        assert!(validate(&unknown).is_err());

        let mut bad_property = ToolCall {
            name: "adjust".into(),
            arguments: serde_json::json!({ "property": "sparkle", "amount": 20 }),
        };
        assert!(normalise(&mut bad_property).is_err());

        let good = ToolCall {
            name: "adjust".into(),
            arguments: serde_json::json!({ "property": "exposure", "amount": 25 }),
        };
        assert!(validate(&good).is_ok());
    }

    #[test]
    fn rejects_a_missing_or_mistyped_argument() {
        let missing = ToolCall {
            name: "adjust".into(),
            arguments: serde_json::json!({ "property": "contrast" }),
        };
        assert!(validate(&missing).is_err());

        let mistyped = ToolCall {
            name: "adjust".into(),
            arguments: serde_json::json!({ "property": "contrast", "amount": "a lot" }),
        };
        assert!(validate(&mistyped).is_err());
    }
}
